import math
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


_EPS = 1e-6

def _parse_float_list(values, default: Optional[List[float]] = None, min_value: Optional[float] = None) -> List[float]:
    if values is None or values == '':
        return list(default or [])
    if isinstance(values, (list, tuple)):
        raw = values
    else:
        raw = [v for v in re.split(r'[ ,;]+', str(values).strip()) if v]
    out = []
    for v in raw:
        try:
            x = float(v)
            if min_value is None or x >= min_value:
                out.append(x)
        except ValueError:
            pass
    return out if out else list(default or [])


def _parse_int_list(values, default: Optional[List[int]] = None, min_value: Optional[int] = None) -> List[int]:
    if values is None or values == '':
        return list(default or [])
    if isinstance(values, (list, tuple)):
        raw = values
    else:
        raw = [v for v in re.split(r'[ ,;]+', str(values).strip()) if v]
    out = []
    for v in raw:
        try:
            x = int(float(v))
            if min_value is None or x >= min_value:
                out.append(x)
        except ValueError:
            pass
    return out if out else list(default or [])


def _parse_periods(periods) -> List[float]:
    return _parse_float_list(periods, default=[24.0, 168.0, 12.0, 48.0], min_value=2.0)


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.exp(x) - 1.0)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        k = self.kernel_size
        left = (k - 1) // 2
        right = k - 1 - left
        x_t = x.permute(0, 2, 1)
        x_pad = F.pad(x_t, (left, right), mode='replicate')
        trend = F.avg_pool1d(x_pad, kernel_size=k, stride=1)
        return trend.permute(0, 2, 1)


class ForecastLinearHead(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, individual: bool = False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.individual = bool(individual)
        if self.individual:
            self.weight = nn.Parameter(torch.empty(self.channels, self.pred_len, self.seq_len))
            self.bias = nn.Parameter(torch.zeros(self.channels, self.pred_len))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.proj = nn.Linear(self.seq_len, self.pred_len)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.permute(0, 2, 1)
        if self.individual:
            out = torch.einsum('bcl,chl->bch', x_c, self.weight) + self.bias.unsqueeze(0)
        else:
            out = self.proj(x_c)
        return out.permute(0, 2, 1)


class SATE(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, scales: List[int],
                 hidden: int = 32, dropout: float = 0.0, individual: bool = False):
        super().__init__()
        if len(scales) == 0:
            scales = [25]
        uniq = []
        for s in scales:
            s = max(1, int(s))
            if s % 2 == 0:
                s += 1
            if s not in uniq:
                uniq.append(s)
        self.scales = uniq
        self.smoothers = nn.ModuleList([MovingAverage(s) for s in self.scales])
        self.router = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(self.scales)),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        if len(self.scales) > 1:
            with torch.no_grad():
                self.router[-1].bias[len(self.scales) // 2] = 0.1
        self.head = ForecastLinearHead(seq_len, pred_len, channels, individual)

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        first = x[:, 0, :]
        last = x[:, -1, :]
        slope = (last - first) / float(max(x.shape[1] - 1, 1))
        volatility = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1) if x.shape[1] > 1 else torch.zeros_like(mean)
        return torch.stack([mean, std, first, last, slope, volatility], dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = torch.stack([ma(x) for ma in self.smoothers], dim=1)
        alpha = torch.softmax(self.router(self._stats(x)), dim=-1)  # 每个通道对应 K 个趋势尺度权重
        trend_hist = torch.einsum('bklc,bck->blc', candidates, alpha)
        trend_pred = self.head(trend_hist)
        return trend_hist, trend_pred, alpha



class SAWDF(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, scales: List[float],
                 kernel_size: int = 25, hidden: int = 64, dropout: float = 0.0,
                 threshold_init: float = 0.01,
                 individual: bool = False, output_scale_init: float = 1.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.scales = [float(s) for s in scales] if len(scales) > 0 else [1.0, 2.0, 4.0, 8.0]
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = max(3, kernel_size)
        self.threshold_raw = nn.Parameter(torch.full((len(self.scales),), _inv_softplus(threshold_init)))

        self.detail_heads = nn.ModuleList([
            ForecastLinearHead(seq_len, pred_len, channels, individual) for _ in self.scales
        ])
        self.n_components = len(self.scales)
        gate_hidden = max(8, int(hidden) // 2)
        self.scale_router = nn.Sequential(
            nn.Linear(6, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.n_components),
        )
        nn.init.zeros_(self.scale_router[-1].weight)
        nn.init.zeros_(self.scale_router[-1].bias)
        self.component_gain = nn.Parameter(torch.zeros(self.n_components))
        self.output_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(output_scale_init)))

        grid = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        self.register_buffer('grid', grid, persistent=False)
        self._last_scale_entropy = None

    def _morlet(self, scale: float, device, dtype) -> torch.Tensor:
        t = self.grid.to(device=device, dtype=dtype)
        sigma = max(float(scale), 1.0)
        u = t / sigma
        wave = torch.cos(5.0 * u) * torch.exp(-0.5 * u ** 2)
        wave = wave - wave.mean()
        wave = wave / torch.sqrt((wave ** 2).sum() + _EPS)
        return wave.view(1, 1, -1)

    def _depthwise_conv(self, x_bc: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        k = kernel.repeat(self.channels, 1, 1)
        pad = self.kernel_size // 2
        x_pad = F.pad(x_bc, (pad, pad), mode='replicate')
        return F.conv1d(x_pad, k, groups=self.channels)

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        first = x[:, 0, :]
        last = x[:, -1, :]
        slope = (last - first) / float(max(x.shape[1] - 1, 1))
        volatility = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1) if x.shape[1] > 1 else torch.zeros_like(mean)
        energy = x.pow(2).mean(dim=1)
        return torch.stack([mean, std, last, slope, volatility, torch.log1p(energy)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bc = x.permute(0, 2, 1)
        component_preds = []

        for i, scale in enumerate(self.scales):
            w = self._morlet(scale, x.device, x.dtype)
            coeff = self._depthwise_conv(x_bc, w)
            local_energy = coeff.detach().abs().mean(dim=-1, keepdim=True)
            tau = F.softplus(self.threshold_raw[i]).to(device=x.device, dtype=x.dtype) * local_energy
            coeff = torch.sign(coeff) * F.relu(coeff.abs() - tau)
            detail_pred = self.detail_heads[i](coeff.permute(0, 2, 1))
            component_preds.append(detail_pred)


        comps = torch.stack(component_preds, dim=1)
        gates = torch.softmax(self.scale_router(self._stats(x)), dim=-1)
        gain = (1.0 + torch.tanh(self.component_gain)).to(device=x.device, dtype=x.dtype)
        weighted = comps * gain.view(1, -1, 1, 1)
        out = torch.einsum('bkhc,bck->bhc', weighted, gates)
        out = F.softplus(self.output_scale_raw).to(device=x.device, dtype=x.dtype) * out

        eps = 1e-8
        entropy = -(gates * torch.log(gates.clamp_min(eps))).sum(dim=-1).mean()
        self._last_scale_entropy = entropy.detach()
        return out


class FPHP(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, periods: List[float],
                 orders: Optional[List[int]] = None, ridge: float = 1e-2,
                 gate_hidden: int = 16, dropout: float = 0.0,
                 gate_bias: float = -2.0, scale_init: float = 1.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.periods = [float(p) for p in periods if float(p) > 2.0]
        if len(self.periods) == 0:
            self.periods = [24.0, 168.0]
        if orders is None or len(orders) == 0:
            orders = [6 for _ in self.periods]
        elif len(orders) == 1 and len(self.periods) > 1:
            orders = orders * len(self.periods)
        elif len(orders) < len(self.periods):
            orders = orders + [orders[-1]] * (len(self.periods) - len(orders))
        self.orders = [max(1, int(o)) for o in orders[:len(self.periods)]]
        self.ridge = float(ridge)

        self._basis_names = []
        all_freqs = []
        for i, (period, order) in enumerate(zip(self.periods, self.orders)):
            hist_basis, fut_basis, freqs = self._build_single_basis(period, order)
            h_name = f'basis_hist_{i}'
            f_name = f'basis_future_{i}'
            self.register_buffer(h_name, hist_basis, persistent=False)
            self.register_buffer(f_name, fut_basis, persistent=False)
            self._basis_names.append((h_name, f_name))
            all_freqs.append(freqs)
        if len(all_freqs) > 0:
            self.register_buffer('basis_freqs', torch.cat(all_freqs, dim=0), persistent=False)
        else:
            self.register_buffer('basis_freqs', torch.zeros(1), persistent=False)
        self.basis_dim = int(self.basis_freqs.numel())

        self.gate = nn.Sequential(
            nn.Linear(5, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        try:
            nn.init.constant_(self.gate[-1].bias, float(gate_bias))
        except Exception:
            pass
        self.scale_raw = nn.Parameter(torch.tensor(_inv_softplus(scale_init)))
        self._last_gate = None
        self._last_coef = None
        self._last_hist_fit = None
        self._last_r2 = None

    def _build_single_basis(self, period: float, order: int):
        t_hist = torch.arange(self.seq_len, dtype=torch.float32)
        t_future = torch.arange(self.pred_len, dtype=torch.float32) + float(self.seq_len)
        hist_cols = []
        fut_cols = []
        freqs = []
        max_order = max(1, int(math.floor(period / 2.0)))
        nyquist = 0.5
        for q in range(1, min(int(order), max_order) + 1):
            f = float(q) / float(period)
            if f >= nyquist:
                continue
            angle_h = 2.0 * math.pi * f * t_hist
            angle_f = 2.0 * math.pi * f * t_future
            hist_cols.extend([torch.cos(angle_h), torch.sin(angle_h)])
            fut_cols.extend([torch.cos(angle_f), torch.sin(angle_f)])
            freqs.extend([f, f])
        if len(hist_cols) == 0:
            hist = torch.zeros(self.seq_len, 1)
            fut = torch.zeros(self.pred_len, 1)
            freqs = [0.0]
        else:
            hist = torch.stack(hist_cols, dim=1)
            fut = torch.stack(fut_cols, dim=1)
            # 使用历史基函数的统计量归一化，并将同样的变换应用到未来基函数。
            mean = hist.mean(dim=0, keepdim=True)
            std = hist.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            hist = (hist - mean) / std
            fut = (fut - mean) / std
        return hist, fut, torch.tensor(freqs, dtype=torch.float32)

    def _fit_one_expert(self, x: torch.Tensor, expert_idx: int):
        bh_name, bf_name = self._basis_names[expert_idx]
        bh = getattr(self, bh_name).to(device=x.device, dtype=x.dtype)
        bf = getattr(self, bf_name).to(device=x.device, dtype=x.dtype)
        b = x.shape[0]
        d = bh.shape[1]
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        gram = bh.transpose(0, 1).matmul(bh) + self.ridge * eye
        rhs = torch.einsum('ld,blc->bdc', bh, x)
        coef = torch.linalg.solve(gram.unsqueeze(0).expand(b, -1, -1), rhs)
        hist_fit = torch.einsum('ld,bdc->blc', bh, coef)
        future_raw = torch.einsum('hd,bdc->bhc', bf, coef)
        return future_raw, hist_fit, coef

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        future_terms = []
        hist_terms = []
        coef_terms = []
        for i in range(len(self.periods)):
            future_raw_i, hist_fit_i, coef_i = self._fit_one_expert(x, i)
            future_terms.append(future_raw_i)
            hist_terms.append(hist_fit_i)
            coef_terms.append(coef_i)

        if len(future_terms) == 0:
            future_raw = torch.zeros(x.shape[0], self.pred_len, self.channels, device=x.device, dtype=x.dtype)
            hist_fit = torch.zeros_like(x)
            coef = torch.zeros(x.shape[0], 1, self.channels, device=x.device, dtype=x.dtype)
        else:
            future_raw = torch.stack(future_terms, dim=0).sum(dim=0)
            hist_fit = torch.stack(hist_terms, dim=0).sum(dim=0)
            coef = torch.cat(coef_terms, dim=1)

        input_energy = x.pow(2).mean(dim=1)
        fit_energy = hist_fit.pow(2).mean(dim=1)
        err_energy = (x - hist_fit).pow(2).mean(dim=1)
        coef_energy = coef.pow(2).mean(dim=1)
        corr = (x * hist_fit).mean(dim=1) / torch.sqrt(input_energy * fit_energy + _EPS)
        r2 = 1.0 - err_energy / (input_energy + _EPS)
        feats = torch.stack([
            torch.log1p(input_energy),
            torch.log1p(fit_energy),
            torch.log1p(err_energy),
            corr.clamp(-1.0, 1.0),
            torch.log1p(coef_energy),
        ], dim=-1)
        gate = torch.sigmoid(self.gate(feats)).permute(0, 2, 1)
        scale = F.softplus(self.scale_raw)
        pred = scale * gate * future_raw

        self._last_r2 = r2.detach()
        return pred, gate, coef, hist_fit


class ACF(nn.Module):
    def __init__(self, pred_len: int, hidden: int = 32, dropout: float = 0.0,
                 horizon_aware: bool = False, horizon_dim: int = 8):
        super().__init__()
        self.pred_len = int(pred_len)
        self.horizon_aware = bool(horizon_aware)
        self.horizon_dim = int(horizon_dim)
        in_dim = 12 + (self.horizon_dim if self.horizon_aware else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        if self.horizon_aware:
            self.horizon_emb = nn.Parameter(torch.randn(self.pred_len, self.horizon_dim) * 0.02)
        else:
            self.register_parameter('horizon_emb', None)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _history_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        first = x[:, 0, :]
        last = x[:, -1, :]
        slope = (last - first) / float(max(x.shape[1] - 1, 1))
        if x.shape[1] > 1:
            volatility = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1)
        else:
            volatility = torch.zeros_like(mean)
        log_energy = torch.log1p(x.pow(2).mean(dim=1))
        return mean, std, slope, volatility, log_energy

    def forward(self, x: torch.Tensor, residual_1: torch.Tensor, residual_2: torch.Tensor,
                trend_pred: torch.Tensor, seasonal_pred: torch.Tensor, wavelet_pred: torch.Tensor,
                seasonal_gate: Optional[torch.Tensor] = None,
                seasonal_r2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, slope, volatility, log_energy = self._history_stats(x)
        r1_energy = torch.log1p(residual_1.pow(2).mean(dim=1))
        r2_energy = torch.log1p(residual_2.pow(2).mean(dim=1))
        trend_rms = torch.log1p(trend_pred.pow(2).mean(dim=1))
        seasonal_rms = torch.log1p(seasonal_pred.pow(2).mean(dim=1))
        wavelet_rms = torch.log1p(wavelet_pred.pow(2).mean(dim=1))
        if seasonal_gate is None:
            sg = torch.zeros_like(mean)
        else:
            sg = seasonal_gate.squeeze(1)
        if seasonal_r2 is None:
            sr2 = torch.zeros_like(mean)
        else:
            sr2 = seasonal_r2
        feat = torch.stack([
            mean,
            std,
            slope,
            volatility,
            log_energy,
            r1_energy,
            r2_energy,
            sg,
            sr2,
            trend_rms,
            seasonal_rms,
            wavelet_rms,
        ], dim=-1)

        if not self.horizon_aware:
            gates = 2.0 * torch.sigmoid(self.net(feat))
            g_t = gates[..., 0].unsqueeze(1)
            g_s = gates[..., 1].unsqueeze(1)
            g_w = gates[..., 2].unsqueeze(1)
        else:
            b, c, _ = feat.shape
            h = self.pred_len
            feat_h = feat.unsqueeze(1).expand(b, h, c, feat.shape[-1])
            emb = self.horizon_emb.to(device=x.device, dtype=x.dtype).view(1, h, 1, self.horizon_dim)
            emb = emb.expand(b, h, c, self.horizon_dim)
            logits = self.net(torch.cat([feat_h, emb], dim=-1))  # 加入预测步嵌入后的门控 logits [B, H, C, 3]
            gates = 2.0 * torch.sigmoid(logits)
            g_t = gates[..., 0]
            g_s = gates[..., 1]
            g_w = gates[..., 2]
        return g_t, g_s, g_w



class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.dropout_p = float(getattr(args, 'dropout', 0.0))
        self.use_revin = bool(getattr(args, 'revin', 1))
        self.individual = bool(getattr(args, 'individual', 0))
        self.ablate_trend = bool(getattr(args, 'ablate_trend', 0))
        self.ablate_irregular = bool(getattr(args, 'ablate_irregular', 0))
        self.ablate_seasonal = bool(getattr(args, 'ablate_seasonal', 0))
        self.trend_residual_detach = bool(getattr(args, 'trend_residual_detach', 1))
        self.use_fusion_gate = bool(getattr(args, 'fusion_gate', 1))
        self.use_horizon_gate = bool(getattr(args, 'horizon_gate', 0))

        if self.use_revin:
            self.revin_layer = RevIN(self.channels, affine=bool(getattr(args, 'affine', 0)),
                                     subtract_last=bool(getattr(args, 'subtract_last', 0)))

        trend_scales = _parse_int_list(
            getattr(args, 'trend_scales', ''),
            default=[13, 25, 49, 97, 169],
            min_value=1,
        )
        self.trend_branch = SATE(
            self.seq_len, self.pred_len, self.channels,
            scales=trend_scales,
            hidden=int(getattr(args, 'trend_router_hidden', 32)),
            dropout=self.dropout_p,
            individual=self.individual,
        )

        wavelet_scales = _parse_float_list(getattr(args, 'wavelet_scales', '1,2,4,8'), default=[1, 2, 4, 8], min_value=0.5)
        self.residual_branch = SAWDF(
            self.seq_len, self.pred_len, self.channels,
            scales=wavelet_scales,
            kernel_size=int(getattr(args, 'wavelet_kernel_size', 25)),
            hidden=int(getattr(args, 'wavelet_hidden', 64)),
            dropout=self.dropout_p,
            threshold_init=float(getattr(args, 'wavelet_threshold_init', 0.01)),
            individual=self.individual,
            output_scale_init=float(getattr(args, 'wavelet_output_scale_init', 1.0)),
        )

        periods = _parse_periods(getattr(args, 'periods', '24,168,12,48'))
        seasonal_orders = _parse_int_list(getattr(args, 'seasonal_orders', ''), default=[6, 4, 3, 3], min_value=1)
        self.seasonal_branch = FPHP(
            self.seq_len, self.pred_len, self.channels,
            periods=periods,
            orders=seasonal_orders,
            ridge=float(getattr(args, 'seasonal_ridge', 1e-2)),
            gate_hidden=int(getattr(args, 'seasonal_gate_hidden', 16)),
            dropout=self.dropout_p,
            gate_bias=float(getattr(args, 'seasonal_gate_bias', -2.0)),
            scale_init=float(getattr(args, 'seasonal_scale_init', 1.0)),
        )

        self.branch_fusion_gate = ACF(
            pred_len=self.pred_len,
            hidden=int(getattr(args, 'fusion_gate_hidden', 32)),
            dropout=self.dropout_p,
            horizon_aware=self.use_horizon_gate,
            horizon_dim=int(getattr(args, 'horizon_gate_dim', 8)),
        ) if self.use_fusion_gate else None

        self._last_components = {}

    def _zero_pred(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.pred_len, self.channels, device=x.device, dtype=x.dtype)

    def forward(self, x_enc):
        if self.use_revin:
            x = self.revin_layer(x_enc, 'norm')
        else:
            x = x_enc

        if self.ablate_trend:
            trend_hist = torch.zeros_like(x)
            trend_pred = self._zero_pred(x)
        else:
            trend_hist, trend_pred, _ = self.trend_branch(x)

        trend_for_residual = trend_hist.detach() if self.trend_residual_detach else trend_hist
        residual_1 = x - trend_for_residual

        if self.ablate_seasonal:
            seasonal_pred = self._zero_pred(x)
            seasonal_gate = torch.zeros(x.shape[0], 1, self.channels, device=x.device, dtype=x.dtype)
        else:
            seasonal_pred, seasonal_gate, _, _ = self.seasonal_branch(residual_1)


        residual_2 = residual_1

        if self.ablate_irregular:
            irregular_pred = self._zero_pred(x)
        else:
            irregular_pred = self.residual_branch(residual_2)

        if self.use_fusion_gate and self.branch_fusion_gate is not None:
            seasonal_r2 = getattr(self.seasonal_branch, '_last_r2', None)
            g_t, g_s, g_w = self.branch_fusion_gate(
                x=x,
                residual_1=residual_1,
                residual_2=residual_2,
                trend_pred=trend_pred,
                seasonal_pred=seasonal_pred,
                wavelet_pred=irregular_pred,
                seasonal_gate=seasonal_gate,
                seasonal_r2=seasonal_r2,
            )
            out_norm = g_t * trend_pred + g_s * seasonal_pred + g_w * irregular_pred
        else:
            g_t = torch.ones_like(trend_pred)
            g_s = torch.ones_like(seasonal_pred)
            g_w = torch.ones_like(irregular_pred)
            out_norm = trend_pred + irregular_pred + seasonal_pred

        self._last_components = {
            'trend_pred': trend_pred.detach(),
            'seasonal_pred': seasonal_pred.detach(),
            'wavelet_pred': irregular_pred.detach(),
            'seasonal_gate': seasonal_gate.detach(),
            'fusion_gate_trend': g_t.detach(),
            'fusion_gate_seasonal': g_s.detach(),
            'fusion_gate_wavelet': g_w.detach(),
            'wavelet_scale_entropy': getattr(self.residual_branch, '_last_scale_entropy', None),
        }


        if self.use_revin:
            out = self.revin_layer(out_norm, 'denorm')
        else:
            out = out_norm
        return out


    @torch.no_grad()
    def get_last_diagnostics(self, target: torch.Tensor = None):
        if not self._last_components:
            return {}

        trend = self._last_components['trend_pred']
        seasonal = self._last_components['seasonal_pred']
        wavelet = self._last_components['wavelet_pred']
        gate = self._last_components['seasonal_gate']
        g_t = self._last_components.get('fusion_gate_trend', torch.ones_like(trend))
        g_s = self._last_components.get('fusion_gate_seasonal', torch.ones_like(seasonal))
        g_w = self._last_components.get('fusion_gate_wavelet', torch.ones_like(wavelet))
        eps = 1e-8

        diag = {
            'trend_rms': trend.pow(2).mean().sqrt().item(),
            'seasonal_rms': seasonal.pow(2).mean().sqrt().item(),
            'wavelet_residual_rms': wavelet.pow(2).mean().sqrt().item(),
            'seasonal_gate_mean': gate.mean().item(),
            'fusion_gate_trend_mean': g_t.mean().item(),
            'fusion_gate_seasonal_mean': g_s.mean().item(),
            'fusion_gate_wavelet_mean': g_w.mean().item(),
        }

        scale_entropy = self._last_components.get('wavelet_scale_entropy', None)
        if scale_entropy is not None:
            diag['wavelet_scale_entropy'] = scale_entropy.item()

        r2 = getattr(self.seasonal_branch, '_last_r2', None)
        if r2 is not None:
            diag['seasonal_fit_r2_mean'] = r2.mean().item()

        if seasonal.numel() > 0 and wavelet.numel() > 0:
            s0 = seasonal - seasonal.mean(dim=1, keepdim=True)
            w0 = wavelet - wavelet.mean(dim=1, keepdim=True)
            sw_corr = (s0 * w0).mean(dim=1) / torch.sqrt(
                s0.pow(2).mean(dim=1) * w0.pow(2).mean(dim=1) + eps
            )
            diag['seasonal_wavelet_corr2'] = sw_corr.pow(2).mean().item()

        return {k: float(v) for k, v in diag.items()}

    @torch.no_grad()
    def learned_frequencies(self):
        return self.seasonal_branch.basis_freqs.detach().cpu()
