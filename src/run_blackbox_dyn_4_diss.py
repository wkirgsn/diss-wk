"""Perform Grid search on dynamic black box models"""

import argparse
from mawkutils.experiments import evaluate_timeseries_nn
from mawkutils.data import DBManager, ChunkedSmoothKaggleDataSet, add_mov_stats
import torch

DEBUG = False
MODEL = ['lstm', 'gru', 'tcn']
N_BATCHES = 4
CHUNK_SIZE = 7200  # 7200 samples = 1 hour (0.5 Hz)
N_JOBS = 1 if DEBUG else 10
TBPTT = int(2*10*60)  # 10 min
CV = '1fold_static_diss'
FE_EXTENSION = ['plain', 'basic', 'extensive']
EWM_FLAGS = [(False, False), (True, False), (True, True)]
IS_RESIDUAL = [False, True]
SPANS = [1320, 3360, 6360, 9480]
N_EPOCHS = 160
HALFS_AT = [80, 120]
LR = 1e-3
N_SEEDS = 10

# grid search for GRU, LSTM, and TCN over
#   EWMA and EWMS, only EWMAs or nothing
#    FE: plain, basic, extensive
#    Residual connections or not

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train temporal NNs')
    parser.add_argument('-t', '--tag', default=None, required=False,
                        help='an identifier/tag/comment for the trials')
    args = parser.parse_args()

    for fe_scheme in FE_EXTENSION:
        for ewma_flag, ewms_flag in EWM_FLAGS:
            ds = ChunkedSmoothKaggleDataSet(chunk_size=CHUNK_SIZE, cv=CV)
            ds.featurize(scheme=fe_scheme)

            if any((ewma_flag, ewms_flag)):
                ds.data = add_mov_stats(ds.data, input_feats=ds.input_cols, pid=ds.pid, spans=SPANS,
                                        add_ewma=ewma_flag, add_ewms=ewms_flag)

                ds.input_cols = [c for c in ds.data if c not in ds.target_cols + [ds.pid]]
                # rearrange
                ds.data = ds.data.loc[:, ds.input_cols + [ds.pid] + ds.target_cols]

            ds.normalize()
            for is_residual in IS_RESIDUAL:
                layer_cfg = {'residual': is_residual, 'f': [{'units': 16}, {'units': 4}]}

                for mdl_t in MODEL:
                    if mdl_t == 'tcn':
                        tbptt = CHUNK_SIZE
                    else:
                        tbptt = TBPTT
                    evaluate_timeseries_nn(cv_method=CV, model_tag=mdl_t, n_epochs=N_EPOCHS, n_batches=N_BATCHES,
                                           tbptt_size=tbptt, run_tag=(args.tag or "") + f" fe_{fe_scheme}",
                                           opt_func=torch.optim.NAdam, lr=LR, debug=DEBUG, n_jobs=N_JOBS, n_seeds=N_SEEDS,
                                           ds=ds, half_lr_at=HALFS_AT, layer_cfg_d=layer_cfg)
