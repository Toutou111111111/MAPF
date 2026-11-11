import argparse
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAPF for Long-term Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='MAPF', help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task: M, S, MS')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='time feature frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='DataLoader pin_memory')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label length used by the data provider')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model size
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # trend
    parser.add_argument('--trend_scales', type=str, default='13,25,49,97,169', help='multi-scale moving-average kernels for adaptive trend')
    parser.add_argument('--trend_router_hidden', type=int, default=32, help='hidden width of adaptive trend router')
    parser.add_argument('--trend_residual_detach', type=int, default=1, help='1: stop gradients through the trend estimate when forming the residual; 0: allow gradients')

    # Morlet-like wavelet
    parser.add_argument('--wavelet_scales', type=str, default='1,2,4,8', help='scales of the Morlet-like filters')
    parser.add_argument('--wavelet_kernel_size', type=int, default=25, help='filter length; even values are rounded up to odd')
    parser.add_argument('--wavelet_hidden', type=int, default=64, help='base width for the wavelet scale router; actual hidden width is max(8, value // 2)')
    parser.add_argument('--wavelet_threshold_init', type=float, default=0.01, help='initial soft-threshold ratio')
    parser.add_argument('--wavelet_output_scale_init', type=float, default=1.0, help='initial output scale of wavelet residual branch')

    # harmonic
    parser.add_argument('--periods', type=str, default='24,168,12,48', help='periods in sampling steps, e.g. "24,168"')
    parser.add_argument('--seasonal_orders', type=str, default='6,4,3,3', help='harmonic orders for each period')
    parser.add_argument('--seasonal_ridge', type=float, default=0.01, help='ridge regularization strength for harmonic fitting')
    parser.add_argument('--seasonal_gate_hidden', type=int, default=16, help='seasonal reliability gate hidden width')
    parser.add_argument('--seasonal_gate_bias', type=float, default=-2.0, help='initial seasonal gate bias; negative is conservative')
    parser.add_argument('--seasonal_scale_init', type=float, default=1.0, help='initial seasonal output scale')

    # gates
    parser.add_argument('--fusion_gate', type=int, default=1, help='1: weight each path with an independent 2*sigmoid gate; 0: add path forecasts directly')
    parser.add_argument('--fusion_gate_hidden', type=int, default=32, help='hidden width of branch fusion gate')
    parser.add_argument('--horizon_gate', type=int, default=0, help='1: use separate path gates for each prediction step and variable; 0: share gates across prediction steps')
    parser.add_argument('--horizon_gate_dim', type=int, default=8, help='horizon embedding dimension for horizon-aware fusion gate')

    # ablations
    parser.add_argument('--ablate_seasonal', type=int, default=0, help='remove harmonic')
    parser.add_argument('--ablate_irregular', type=int, default=0, help='remove wavelet')
    parser.add_argument('--ablate_trend', type=int, default=0, help='remove trend')

    # normalization / heads
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN affine')
    parser.add_argument('--subtract_last', type=int, default=0, help='RevIN subtract last')
    parser.add_argument('--individual', type=int, default=0, help='individual linear head; keep 0 for ETTh1')

    # embedding/time feature compatibility
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float , default=0.0, help='weight decay')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='cosine', help='learning rate adjustment: none/type1/type2/cosine')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum lr for cosine schedule')
    parser.add_argument('--use_amp', action='store_true', default=False, help='automatic mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip')
    parser.add_argument('--log_interval', type=int, default=100, help='training log interval')
    parser.add_argument('--print_diagnostics', type=int, default=1, help='print once-per-epoch diagnostics if exp_main supports it')
    parser.add_argument('--save_pdf', action='store_true', default=False, help='save visual PDFs during testing')
    parser.add_argument('--save_outputs',action='store_true',default=False,help='save pred.npy and true.npy after testing')

    parser.add_argument('--save_input', action='store_true', default=False, help='save test input x.npy')
    parser.add_argument('--do_predict', action='store_true', default=False, help='predict unseen future data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    def _compact(s):
        return str(s).replace(',', '-').replace(' ', '').replace('.', 'p')


    def make_setting(ii):
        return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_ts{}_per{}_so{}_ws{}_wks{}_wt{}_wd{}_sg{}_fg{}_fgh{}_hg{}_dp{}_lr{}_seed{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            _compact(args.trend_scales),
            _compact(args.periods),
            _compact(args.seasonal_orders),
            _compact(args.wavelet_scales),
            _compact(args.wavelet_kernel_size),
            str(args.wavelet_threshold_init).replace('.', 'p'),
            str(args.wavelet_output_scale_init).replace('.', 'p'),
            str(args.seasonal_gate_bias).replace('.', 'p').replace('-', 'm'),
            args.fusion_gate,
            args.fusion_gate_hidden,
            args.horizon_gate,
            _compact(args.dropout),
            _compact(args.learning_rate),
            args.random_seed,
            ii,
        )


    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            setting = make_setting(ii)
            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = make_setting(ii)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
