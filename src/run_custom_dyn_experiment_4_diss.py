"""Perform Grid search on dynamic black box models"""

import argparse
from mawkutils.experiments import evaluate_timeseries_nn
from mawkutils.data import DBManager, ChunkedSmoothKaggleDataSet, add_mov_stats
import torch

DEBUG = False
MODEL = [  # 'lstm', 'gru',
    'tnn', #'tcn',
    #'single_sub_tnn',
    #'expleuler'
    # 'capless_tnn'
]
N_BATCHES = 4
CHUNK_SIZE = 5870  # 7200 samples = 1 hour (0.5 Hz)
TBPTT = int(558)  # 10 min
CV = '1fold_static_diss'
FE_EXTENSION = [#  'plain', 
                'basic',
    #'extensive'
    ]
EWM_FLAGS = [  # (False, False),
    (False, False),
    #(True, True)
]
IS_RESIDUAL = [  # False,
    False]
SPANS = [1320, 3360, 6360, 9480]
N_EPOCHS = 160
HALFS_AT = [80, 120]
LR = 17.9e-3
N_SEEDS = 30
N_JOBS = 1 if DEBUG else 8


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
                #layer_cfg = {'residual': is_residual, 'g': [{'units': 32}, {'units': 4}]}
                conductances_to_drop = ["pm,stator_winding", "stator_winding,ambient", "stator_winding,coolant",
                              'pm,stator_yoke', 'pm,coolant', 'stator_tooth,ambient', 'stator_tooth,coolant'
                              ]
                layer_cfg = {
                   'p': [{'units': 2, 'activation': 'tanh'},
                         {'units': 4, 'activation': 'sigmoid'}],
                   'g': [{'units': 2, 'activation': 'relu'},
                         {'units': int(0.5 * 6*5) - len(conductances_to_drop), 'activation': 'biased_elu'}],
                   'drop_g': conductances_to_drop,

                }
                """layer_cfg = {
                    'f': [{'units': 32, 'activation': 'relu'},
                          {'units': len(ds.target_cols), 'activation': 'tanh'}],
                }"""
                for mdl_t in MODEL:
                    evaluate_timeseries_nn(cv_method=CV, model_tag=mdl_t, n_epochs=N_EPOCHS, n_batches=N_BATCHES,
                                           tbptt_size=TBPTT,
                                           run_tag=(args.tag or "") + f" fe_{fe_scheme}",
                                           opt_func=torch.optim.NAdam, lr=LR, debug=DEBUG, n_jobs=N_JOBS, n_seeds=N_SEEDS,
                                           ds=ds, half_lr_at=HALFS_AT, layer_cfg_d=layer_cfg)
