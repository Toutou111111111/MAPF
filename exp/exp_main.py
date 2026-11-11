import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
    MAPF
)
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            'MAPF': MAPF,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 0.0),
        )

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'mse')).lower()
        if loss_name == 'mae':
            return nn.L1Loss()
        if loss_name in ['smooth', 'smoothl1', 'smooth_l1']:
            return nn.SmoothL1Loss()
        return nn.MSELoss()

    def _model_module(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _forward_model(self, batch_x):
        return self.model(batch_x)

    def _slice_prediction_and_target(self, outputs, batch_y):
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, batch_y

    def _get_epoch_diagnostics(self, batch_y_use=None):
        module = self._model_module()
        if hasattr(module, 'get_last_diagnostics'):
            try:
                return module.get_last_diagnostics(batch_y_use)
            except TypeError:
                return module.get_last_diagnostics()
        return None

    def _format_diagnostics(self, diag):
        if not diag:
            return ''

        def _append(parts, label, value):
            try:
                parts.append(f'{label}: {float(value):.6g}')
            except Exception:
                parts.append(f'{label}: {value}')

        order = [
            ('trend_rms', 'trend_rms'),
            ('seasonal_rms', 'seasonal_rms'),
            ('wavelet_residual_rms', 'wavelet_residual_rms'),
            ('wavelet_scale_entropy', 'wavelet_scale_entropy'),
            ('seasonal_gate_mean', 'seasonal_gate_mean'),
            ('seasonal_fit_r2_mean', 'seasonal_fit_r2_mean'),
            ('seasonal_wavelet_corr2', 'seasonal_wavelet_corr2'),
            ('fusion_gate_trend_mean', 'fusion_gate_trend_mean'),
            ('fusion_gate_seasonal_mean', 'fusion_gate_seasonal_mean'),
            ('fusion_gate_wavelet_mean', 'fusion_gate_wavelet_mean'),
        ]

        parts = []
        printed = set()
        for key, label in order:
            if key in diag:
                _append(parts, label, diag[key])
                printed.add(key)
        for key, value in diag.items():
            if key not in printed:
                _append(parts, key, value)
        return ' | '.join(parts)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_model(batch_x)
                outputs, batch_y_use = self._slice_prediction_and_target(outputs, batch_y)
                loss = criterion(outputs, batch_y_use)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            epoch_diag = None

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x)
                        outputs, batch_y_use = self._slice_prediction_and_target(outputs, batch_y)
                        loss = criterion(outputs, batch_y_use)
                    scaler.scale(loss).backward()
                    if getattr(self.args, 'grad_clip', 0) > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, 'grad_clip', 0))
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_model(batch_x)
                    outputs, batch_y_use = self._slice_prediction_and_target(outputs, batch_y)
                    loss = criterion(outputs, batch_y_use)

                    loss.backward()
                    if getattr(self.args, 'grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, 'grad_clip', 0))
                    model_optim.step()

                train_loss.append(loss.item())

                if epoch_diag is None and int(getattr(self.args, 'print_diagnostics', 1)) == 1:
                    epoch_diag = self._get_epoch_diagnostics(batch_y_use)

                log_interval = int(getattr(self.args, 'log_interval', 100))
                if (i + 1) % log_interval == 0:
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}')
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                f'Epoch: {epoch + 1}, Steps: {train_steps} | '
                f'Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}'
            )

            if epoch_diag:
                print(f'Epoch: {epoch + 1} Diagnostics | {self._format_diagnostics(epoch_diag)}')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                    map_location=self.device,
                )
            )

        save_outputs = bool(getattr(self.args, 'save_outputs', False))
        save_input = bool(getattr(self.args, 'save_input', False))
        save_pdf = bool(getattr(self.args, 'save_pdf', False))

        preds, trues = [], []
        inputx = [] if save_input else None

        folder_path = os.path.join('./test_results', setting)
        if save_pdf:
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_model(batch_x)
                outputs, batch_y_use = self._slice_prediction_and_target(outputs, batch_y)

                pred = outputs.detach().cpu().numpy()
                true = batch_y_use.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if save_input:
                    inputx.append(batch_x.detach().cpu().numpy())

                if save_pdf and i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')

        with open('result.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write(f'mse:{mse}, mae:{mae}')
            f.write('\n\n')

        np.save(
            os.path.join(folder_path, 'metrics.npy'),
            np.array([mae, mse, rmse, mape, mspe, rse, corr]),
        )

        if save_outputs:
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
            np.save(os.path.join(folder_path, 'true.npy'), trues)

        if save_input:
            inputx = np.concatenate(inputx, axis=0)
            np.save(os.path.join(folder_path, 'x.npy'), inputx)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _, _, _ in pred_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self._forward_model(batch_x)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        if pred_data.scale:
            original_shape = preds.shape
            preds = pred_data.inverse_transform(
                preds.reshape(-1, original_shape[-1])
            ).reshape(original_shape)
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
