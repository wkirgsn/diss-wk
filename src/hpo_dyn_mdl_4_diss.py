import optuna
import platform
import argparse
import sshtunnel
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from mawkutils.data import DBManager, ChunkedSmoothKaggleDataSet, add_mov_stats
from mawkutils.validation import calc_metrics
from mawkutils.topology import ACTIVATION_FUNCS
from mawkutils.experiments import evaluate_timeseries_nn


DEBUG = False
FORCE_LOCAL_STORAGE = True  # should always be true for anyone trying this codebase
DB_NAME = 'optuna'
STUDY_NAME = 'study_3_4_diss'
DEBUG_STUDY_NAME = 'study_debug_4_diss'
PROC_DATA_PATH = Path.cwd().parent / 'data' / 'processed'
ALLOWED_MODEL_TAGS = ('tnn', 'gru', 'lstm', 'tcn')

# only for UPB server necessary
SERVER_LOCAL_PORT2PSQL = 5432
PC2_LOCAL_PORT2PSQL = 35219


OPTIMIZERS = {'adam': torch.optim.Adam, 'nadam': torch.optim.NAdam,
              'adamax': torch.optim.Adamax, 'sgd': torch.optim.SGD,
              'rmsprop': torch.optim.RMSprop}
N_JOBS = 1 if DEBUG else 4
N_SEEDS = 11
CV = 'hpo_1fold_diss'
N_EPOCHS = 160
HALFS_AT = [80, 120]
SPANS = [1320, 3360, 6360, 9480]


def optuna_optimize(objective, n_trials=1, sampler=None, study_name='dummy'):
    if n_trials < 0:
        n_trials = None  # conduct trials indefinitely

    node = platform.uname().node

    if FORCE_LOCAL_STORAGE or \
        (node not in DBManager.HOSTS_WITH_DIRECT_PORT_ACCESS + DBManager.LEA_VPN_NODES and
         not node.lower().startswith(("lea-", "cn-", "node", "n2cn"))):
        # local sqlite storage for optuna
        print("## Optuna optimize on local storage ##")
        study = optuna.create_study(
            storage=f'sqlite:///{DBManager.LOCAL_SQLITE_PATH.resolve() / (DB_NAME + ".sqlite")}',
            sampler=sampler, study_name=study_name, direction="minimize",
            load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)
    else:
        # read db credentials
        with open(f'{os.getenv("HOME")}/creds/optuna_psql', 'r') as f:
            optuna_creds = ':'.join([s.strip() for s in f.readlines() if len(s.strip()) > 0])
        if node in DBManager.HOSTS_WITH_DIRECT_PORT_ACCESS:
            port = SERVER_LOCAL_PORT2PSQL if node == DBManager.SERVER_NAME \
                else PC2_LOCAL_PORT2PSQL

            study = optuna.create_study(
                storage=f'postgresql://{optuna_creds}@localhost:{port}/{DB_NAME}',
                sampler=sampler, study_name=study_name, direction="minimize",
                load_if_exists=True)
            study.optimize(objective, n_trials=n_trials)
        else:
            if node in DBManager.LEA_VPN_NODES or node.lower().startswith("lea-"):
                # we are in LEA VPN
                server_cfg_name = "lea38"
                tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                   SERVER_LOCAL_PORT2PSQL)}
            else:
                # assume we are on a PC2 compute node
                assert node.lower().startswith(("cn-", "n2cn")), \
                    f"The hostname could not be identified: {node}"

                server_cfg_name = 'ln-0002' if node.lower().startswith("cn-") \
                    else "n2login3"
                tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                   PC2_LOCAL_PORT2PSQL),
                           'ssh_username': 'wilhelmk'}
            with sshtunnel.open_tunnel(server_cfg_name, **tun_cfg) as tun:
                study = optuna.create_study(
                    storage=f'postgresql://{optuna_creds}'
                            f'@localhost:{tun.local_bind_port}/{DB_NAME}',
                    sampler=sampler, study_name=study_name, direction="minimize",
                    load_if_exists=True)
                study.optimize(objective, n_trials=n_trials)


def get_model_specific_objective(mdl_tag):

    black_box_mdls = ('gru', 'lstm', 'tcn')

    def objective(trial):
        # training scheme HPs
        chunk_size = trial.suggest_int("chunk_size", 450, 28800)  # 1 hour = 7200 samples (2 Hz)
        n_batches = trial.suggest_int("n_batches", 1, 4)
        if mdl_tag == 'tnn':
            fe_extension_scheme = trial.suggest_categorical("fe_extension", ["plain", "basic", "extensive"])
        elif mdl_tag in black_box_mdls:
            fe_extension_scheme = 'extensive'

        ds = ChunkedSmoothKaggleDataSet(chunk_size=chunk_size, cv=CV)
        ds.featurize(scheme=fe_extension_scheme)

        if mdl_tag in black_box_mdls:
            has_ewma = trial.suggest_categorical("has_ewma", [False, True])
            if has_ewma:
                ds.data = add_mov_stats(ds.data, input_feats=ds.input_cols, pid=ds.pid,
                                        spans=SPANS, add_ewma=has_ewma, add_ewms=False)
                ds.input_cols = [c for c in ds.data if c not in ds.target_cols + [ds.pid]]
                # rearrange
                ds.data = ds.data.loc[:, ds.input_cols + [ds.pid] + ds.target_cols]
        ds.normalize()

        # architecture HPs
        if mdl_tag == 'tnn':
            act_funcs = list(ACTIVATION_FUNCS.keys())
            targets = ds.target_cols
            n_temps = len(ds.temperature_cols)
            last_layer_units = {'p': len(targets), 'g': int(0.5*n_temps*(n_temps-1))}
            layers_cfg_d = {'p': [], 'g': []}
            for k in layers_cfg_d:
                n_layers = trial.suggest_int(f'n_{k}_layers', 0, 3)
                for i in range(n_layers):
                    layers_cfg_d[k].append(
                        {'units': trial.suggest_int(f'n_units_{k}_layer_{i}', 2, 64, log=False),
                         'activation': trial.suggest_categorical(f'act_{k}_layer_{i}', act_funcs),
                         'name': f'{k}_{i}',

                         })
                layers_cfg_d[k].append({
                    'units': last_layer_units[k],
                    'activation': trial.suggest_categorical(f'{k}_out_act', act_funcs),
                    'name': f'{k}_output'
                })
            layers_cfg_d["cap"] = [trial.suggest_float(f"inv_caps_init_gauss_mean_{t}", -20.0, -0.1)
                                   for t in targets]  # caps normal mean (log)
        elif mdl_tag in black_box_mdls:
            is_res = trial.suggest_categorical(f"is_residual", [False, True])
            layers_cfg_d = {'residual': is_res, 'f': []}
            n_layers = trial.suggest_int(f'n_layers', 0, 2)
            for i in range(n_layers):
                layer_cfg = {'units': trial.suggest_int(f'n_units_layer_{i}', 2, 64, log=False),
                             'name': f'f_{i}',
                             }
                if mdl_tag == 'tcn':
                    layer_cfg['kernel_size'] = trial.suggest_int(f'kernel_size_layer_{i}', 2, 7, log=False)
                layers_cfg_d["f"].append(layer_cfg)
            last_layer_cfg = {
                'units': len(ds.target_cols),
                'name': f'f_output'
            }
            if mdl_tag == 'tcn':
                last_layer_cfg['kernel_size'] = trial.suggest_int(f'kernel_size_layer_out', 2, 7, log=False)
                layers_cfg_d["starting_dilation_rate"] = trial.suggest_int(f"starting_dilation_rate", 0, 2)
                layers_cfg_d["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
            layers_cfg_d["f"].append(last_layer_cfg)
            

        # optimization HPs
        tbptt_len = trial.suggest_int('tbptt_len', 4, chunk_size, log=False)
        
        lr = trial.suggest_float('inital_lr', 1e-5, 1, log=True)
        optimizer = trial.suggest_categorical('optimizer', list(OPTIMIZERS.keys()))

        print(f"Start trial with: chunk_size={chunk_size}, n_batches={n_batches}, tbptt={tbptt_len}")
        exp_logs = evaluate_timeseries_nn(cv_method=CV, model_tag=mdl_tag, n_epochs=N_EPOCHS, n_batches=n_batches,
                                          tbptt_size=tbptt_len, run_tag=f"HPO {DEBUG_STUDY_NAME if DEBUG else STUDY_NAME} trial-nr {trial.number}",
                                          opt_func=OPTIMIZERS[optimizer], lr=lr, debug=DEBUG, n_jobs=N_JOBS, n_seeds=N_SEEDS,
                                          layer_cfg_d=layers_cfg_d, ds=ds, half_lr_at=HALFS_AT)

        metrics = [calc_metrics(l['all_predictions_df'].loc[:, ds.target_cols],
                                l['ground_truth'].loc[:, ds.target_cols],
                                target_scale=ds.temperature_scale) for l in exp_logs
                   if not l["all_predictions_df"].isnull().values.any()]  # filter NaNs
        avg_loss = np.median([m['mse'] for m in metrics])
        return avg_loss

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HPO for dynamic models 4 diss')
    parser.add_argument('-m', '--modeltag', default=None, required=True,
                        help=f'an identifier/tag/comment for the model. Must be in {ALLOWED_MODEL_TAGS}')
    parser.add_argument('-n', '--n_trials', default=1, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    model_tag = args.modeltag
    assert model_tag in ALLOWED_MODEL_TAGS, f"{model_tag=} must be in {ALLOWED_MODEL_TAGS}"
    study_name = DEBUG_STUDY_NAME if DEBUG else STUDY_NAME
    study_name += f'_{model_tag}'

    optuna_optimize(get_model_specific_objective(model_tag), study_name=study_name, n_trials=args.n_trials)
