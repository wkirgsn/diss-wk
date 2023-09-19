"""Static Modeling experiments for black box chapter in WK dissertation, 2022"""
import numpy as np
import pandas as pd
import pathlib
import sys
import argparse
import platform
from joblib import Parallel, delayed
import json
import random
from tqdm import trange
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error as mse
#from lightgbm import LGBMRegressor
import torch
from torch import nn
from mawkutils.data import KaggleDataSet, SmoothKaggleDataSet, DBManager, add_mov_stats
from mawkutils.topology import MLP


DEBUG = False
CV = '1fold_no_val_static_diss'
DUMP_PATH = pathlib.Path.cwd().parent / 'data' / 'output' / 'static_blackbox_estimates'
DUMP_PATH.mkdir(exist_ok=True, parents=True)
N_SEEDS = 3 if DEBUG else 10
BATCH_SIZE = 2**8  # only for MLP (2**11)
N_EPOCHS = 100  # only for MLP (400)
EWM_ADJUST = False

FE_SCHEMES = ['plain', 'basic', 'extensive']
EWMA_FLAGS = [False, True]
EWMS_FLAGS = [False, True]
SPANS = [1320, 3360, 6360, 9480]
ZOO_d = {
    #'ols': (LinearRegression, {'n_jobs': -1}),
     "ridge": (Ridge, {}),
    # "lasso": (Lasso, {'alpha': 1e-4}),
    # 'svr': (LinearSVR, {'dual': False, 'loss': 'squared_epsilon_insensitive'}),
    # 'rf': (RandomForestRegressor, {"n_jobs": -1, "max_depth": 10, "n_estimators": 20}),
    # 'et': (ExtraTreesRegressor,  {"n_jobs": -1, "max_depth": 10, "n_estimators": 20}),
    # 'lgbm': (LGBMRegressor, {"n_jobs": -1, "max_depth": 5, "n_estimators": 50}),
    #'histgbm': (HistGradientBoostingRegressor, {"max_depth": 5, "max_iter": 50}),
    # 'mlp': (MLP, {})

}


def get_tree_size(tree):
    """Returns DecisionTree size when planning for deployment on embedded systems 
    (how many floating point numbers need to be stored?)

    example:
    >>> tree_sizes = [get_tree_size(t.tree_) for t in mdl.estimators_]
    >>> print(tree_sizes)
    >>> sum(tree_sizes)
    """
    n_leaves = tree.n_leaves  # constant floating point prediction per leaf
    n_thresholds = tree.node_count - n_leaves  # constant floating point threshold per internal node
    n_features_to_split = n_thresholds  # integer index per internal node, addressing the input feature to split on
    # each leaf is associated with n_target constant predictions
    return int(n_leaves * tree.value.shape[1] + n_thresholds + n_features_to_split)


def get_gbm_tree_size(tree):
    """Returns HistGradientBoosting DecisionTree size when planning for deployment on embedded systems.
    example:
    >>> tree_sizes = [get_gbm_tree_size(t[0]) for t in mdl._predictors
    >>> sum(tree_sizes)
    """
    n_leafs = tree.get_n_leaf_nodes()
    n_total_nodes = tree.nodes.shape[0]
    n_thresholds = n_total_nodes - n_leafs
    return int(n_leafs + 2*n_thresholds)


def get_model_size(mdl, mdl_lbl):
    """Depending on the mdl_lbl, determine the model size"""
    def get_linear_models_size(mdl):
        mdl_size = mdl.coef_.size
        if len(mdl.coef_.shape) > 1:
            mdl_size += mdl.coef_.shape[0]
        return int(mdl_size)
    if mdl_lbl in ('ols', 'ridge', 'lasso', 'svr'):
        return get_linear_models_size(mdl)
    elif mdl_lbl in ('rf', 'et'):
        return sum(get_tree_size(t.tree_) for t in mdl.estimators_)
    elif mdl_lbl in ("histgbm"):
        return sum(get_gbm_tree_size(t[0]) for t in mdl._predictors)
    elif mdl_lbl == 'mlp':
        return int(sum(int(param.nelement()) for param in mdl.parameters()))
    else:
        raise NotImplementedError(mdl_lbl)


class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=False):
        self.column_scale = {c: KaggleDataSet.temperature_scale for c in
                             KaggleDataSet.input_temperature_cols + KaggleDataSet.target_cols}
        self.verbose = verbose
        self.fitted_cols = []

    def fit(self, X, y=None, sample_weight=None):
        assert isinstance(X, pd.DataFrame), "X is no pandas DataFrame"
        for c in X:
            if c not in self.column_scale:
                self.column_scale[c] = X.loc[:, c].abs().max(axis=0)
            self.fitted_cols.append(c)
        if self.verbose:
            print(self.column_scale)
        return self

    def transform(self, X, copy=None):
        assert isinstance(X, pd.DataFrame), "X is no pandas DataFrame"
        cols = X.columns.tolist()
        scale = np.array([self.column_scale[c] for c in cols]).reshape(1, -1)
        return X / scale

    def inverse_transform(self, X, copy=None):
        #assert isinstance(X, pd.DataFrame), "X is no pandas DataFrame"
        scale = np.array([self.column_scale[c] for c in self.fitted_cols]).reshape(1, -1)
        return X * scale





def add_features(data_in, fe_scheme):
    """In this script, the data preprocessing phase is a bit varied wrt the dataset class in mawkutils.data.
    That is why some functions like this one appear again although they already appear similarly in mawktutuks.data.
    For example, for static models in the diss, I also vary the normalization scheme, which is fix otherwise at limit normalization.
    Moreover, feature selection is partly performable here."""
    # TODO: Maybe unify everything into mawkutils.data, even if the varying normalization is special to static models?
    extra_feats = {}
    if fe_scheme == 'basic':
        extra_feats = {'i_s': lambda x: np.sqrt((x['i_d']**2 + x['i_q']**2)),
                       'u_s': lambda x: np.sqrt((x['u_d']**2 + x['u_q']**2))}
    elif fe_scheme == 'extensive':
        extra_feats = {'i_s': lambda x: np.sqrt((x['i_d']**2 + x['i_q']**2)),
                       'u_s': lambda x: np.sqrt((x['u_d']**2 + x['u_q']**2)),
                       'S_el': lambda x: x['i_s'] * x['u_s'],
                       'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] * x['u_q'],
                       'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
                       'S_x_w': lambda x: x['S_el']*x['motor_speed'],
                       }
    return data_in.assign(**extra_feats)


def normalize_data(data_in, input_feats, target_cols, scaler_cls):
    data_out = data_in.copy()
    input_scaler = scaler_cls()
    target_scaler = scaler_cls()
    data_out.loc[:, input_feats] = input_scaler.fit_transform(data_out.loc[:, input_feats])
    data_out.loc[:, target_cols] = target_scaler.fit_transform(data_out.loc[:, target_cols])
    return data_out, input_scaler, target_scaler


SCALERS = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'custom': CustomScaler}


def run_blackbox_static(run_tag, fe_schemes=None, ewma_flags=None, ewms_flags=None, zoo_cfg_d=None,
                        spans=None, scalers_d=None, cv=None, ds=None, debug=None, mlp_cv=None, n_seeds=None,
                        custom_input_feature_whitelist=None):
    "Iterate over many FE scheme variations"
    zoo_d = zoo_cfg_d or ZOO_d
    fe_schemes = fe_schemes or FE_SCHEMES
    ewma_flags = ewma_flags or EWMA_FLAGS
    ewms_flags = ewms_flags or EWMS_FLAGS
    scalers_d = scalers_d or SCALERS
    debug = debug or DEBUG
    cv = cv or CV
    n_seeds = n_seeds or N_SEEDS

    ds = ds or SmoothKaggleDataSet()
    if debug:
        ds.data = pd.concat([df.iloc[:1000, :] for p, df
                             in ds.data.groupby(ds.pid, sort=False)], ignore_index=True)  # crop
    # iterate over models
    for mdl_lbl, (mdl_cls, mdl_kwargs) in zoo_d.items():

        # featurize
        for fe_scheme in fe_schemes:
            data_1 = add_features(ds.data, fe_scheme)

            for has_ewmas in ewma_flags:
                for has_ewmsts in ewms_flags:
                    input_feats = [c for c in data_1 if c not in ds.target_cols + [ds.pid]]
                    if has_ewmas or has_ewmsts:
                        # add EWM features
                        data_2 = add_mov_stats(data_1, input_feats, ds.pid, spans or SPANS,
                                               has_ewmas, has_ewmsts)
                        input_feats = [c for c in data_2 if c not in ds.target_cols + [ds.pid]]
                    else:
                        data_2 = data_1
                    if custom_input_feature_whitelist is not None:
                        # overwrite input feats with a certain selection
                        input_feats = custom_input_feature_whitelist
                    # rearrange
                    data_2 = data_2.loc[:, input_feats + [ds.pid] + ds.target_cols]

                    for norm_scheme_lbl, scaler_cls in scalers_d.items():
                        # normalize
                        data_3, input_scaler, target_scaler =\
                            normalize_data(data_2, input_feats, ds.target_cols, scaler_cls)

                        # split data into sets
                        cv_lbl = cv if mdl_lbl != "mlp" else (mlp_cv or "1fold_static_diss")
                        train_l, val_l, test_l = ds.get_profiles_for_cv(cv_lbl, verbose=False)

                        def run_static_training(rep=0):
                            """Run training with a certain seed"""
                            np.random.seed(rep)
                            random.seed(rep)
                            torch.manual_seed(rep)

                            log = {'loss_trends_train': [[] for _ in train_l],
                                   'loss_trends_val': [[] for _ in val_l],
                                   'models_state_dict': [],
                                   'start_time': pd.Timestamp.now().round(freq='S')
                                   }

                            for fold_i, (train_fold_l, val_fold_l, test_fold_l) in enumerate(zip(train_l, val_l, test_l)):

                                train_data = data_3.loc[data_3.loc[:, ds.pid].isin(
                                    train_fold_l), :].reset_index(drop=True)
                                # shuffle train data
                                if rep > 0:
                                    train_data.index.name = 'idx'
                                    train_data = train_data.sample(frac=1, random_state=rep).reset_index(drop=False)
                                    orig_train_row_idx = train_data.pop('idx')
                                train_pid_s = train_data.pop(ds.pid).reset_index(drop=True)
                                test_data = data_3.loc[data_3.loc[:, ds.pid].isin(
                                    test_fold_l), :].reset_index(drop=True)
                                test_pid_s = test_data.pop(ds.pid).reset_index(drop=True)
                                if len(val_l) != 0:
                                    val_data = data_3.loc[data_3.loc[:, ds.pid].isin(val_fold_l), :]
                                    val_pid_s = val_data.pop(ds.pid)
                                else:
                                    val_data = None

                                # maybe train for each target feature
                                if mdl_lbl in ["svr", "lgbm", "histgbm"]:
                                    train_preds_l = []
                                    test_preds_l = []
                                    mdl_size = 0
                                    sub_model_kwargs = mdl_kwargs.copy()
                                    if mdl_lbl == 'svr':
                                        sub_model_kwargs['random_state'] = rep
                                    for target_lbl in ds.target_cols:
                                        mdl = mdl_cls(**sub_model_kwargs)
                                        # train and predict
                                        mdl.fit(train_data.loc[:, input_feats], train_data.loc[:, target_lbl])
                                        train_pred = pd.Series(
                                            mdl.predict(train_data.loc[:, input_feats]), name=target_lbl)
                                        test_pred = pd.Series(
                                            mdl.predict(test_data.loc[:, input_feats]), name=target_lbl)
                                        train_preds_l.append(train_pred)
                                        test_preds_l.append(test_pred)
                                        mdl_size += get_model_size(mdl, mdl_lbl)
                                    # end_time is somewhat skewed since predictions are included in time
                                    log["end_time"] = pd.Timestamp.now().round(freq='S')
                                    train_pred = pd.DataFrame(target_scaler.inverse_transform(
                                        np.column_stack(train_preds_l)), columns=ds.target_cols).assign(**{ds.pid: train_pid_s})
                                    test_pred = pd.DataFrame(target_scaler.inverse_transform(
                                        np.column_stack(test_preds_l)), columns=ds.target_cols).assign(**{ds.pid: test_pid_s})
                                else:
                                    if mdl_lbl == 'mlp':
                                        mlp_kwargs = mdl_kwargs.copy()
                                        mlp_kwargs['n_inputs'] = len(input_feats)
                                        device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
                                        mdl = mdl_cls(**mlp_kwargs).to(device)
                                        # prepare data
                                        if val_data is not None:
                                            val_tens = torch.from_numpy(
                                                val_data.to_numpy().astype(np.float32)).to(device)
                                        test_tens = torch.from_numpy(test_data.to_numpy().astype(np.float32)).to(device)
                                        n_targets = len(ds.target_cols)
                                        # setup optimizers
                                        loss_fn = nn.MSELoss()
                                        optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)
                                        pbar = trange(N_EPOCHS)
                                        # train
                                        for epoch in pbar:
                                            shuffled_train_data = torch.from_numpy(train_data.sample(frac=1)
                                                                                   .reset_index(drop=True)
                                                                                   .to_numpy().astype(np.float32)).to(device)
                                            for batch_idx in range(np.ceil(len(shuffled_train_data) / BATCH_SIZE).astype(int)):
                                                batch_start, batch_end = batch_idx * \
                                                    BATCH_SIZE, (batch_idx+1)*BATCH_SIZE
                                                x = shuffled_train_data[batch_start:batch_end, :-n_targets]
                                                y = shuffled_train_data[batch_start:batch_end, -n_targets:]
                                                y_hat = mdl(x)
                                                loss = loss_fn(y_hat, y)
                                                # backprop
                                                optimizer.zero_grad()
                                                loss.backward()
                                                optimizer.step()
                                            log['loss_trends_train'][fold_i].append(loss.item())
                                            pbar_str = f'Loss {loss.item():.2e}'
                                            # validation set
                                            if val_data is not None:
                                                with torch.no_grad():
                                                    # TODO: rather do a batchful prediction?
                                                    val_pred = mdl(val_tens[:, :-n_targets])
                                                    val_loss = loss_fn(val_pred, val_tens[:, -n_targets:]).item()
                                                    log['loss_trends_val'][fold_i].append(val_loss)
                                                    pbar_str += f'| val loss {val_loss:.2e}'
                                            pbar.set_postfix_str(pbar_str)
                                        log["end_time"] = pd.Timestamp.now().round(freq='S')
                                        # test
                                        with torch.no_grad():
                                            test_pred = mdl(test_tens[:, :-n_targets]).cpu().detach().numpy()
                                            #test_loss = loss_fn(test_pred, test_data[:, -n_targets:]).item()
                                            test_pred = pd.DataFrame(target_scaler.inverse_transform(
                                                test_pred), columns=ds.target_cols).assign(**{ds.pid: test_pid_s})
                                        train_pred = None
                                        log["models_state_dict"].append(mdl.state_dict())
                                        if 'models_arch' not in log:
                                            log['models_arch'] = json.dumps(mdl.layer_cfg)
                                    elif mdl_lbl == 'lasso':
                                        lasso_kwargs = mdl_kwargs.copy()
                                        lasso_kwargs['random_state'] = rep
                                        mdl = mdl_cls(**lasso_kwargs)
                                        # train and predict
                                        mdl.fit(train_data.loc[:, input_feats], train_data.loc[:, ds.target_cols])
                                        log["end_time"] = pd.Timestamp.now().round(freq='S')
                                        train_pred = pd.DataFrame(target_scaler.inverse_transform(mdl.predict(train_data.loc[:, input_feats])),
                                                                  columns=ds.target_cols).assign(**{ds.pid: train_pid_s})
                                        test_pred = pd.DataFrame(target_scaler.inverse_transform(mdl.predict(test_data.loc[:, input_feats])),
                                                                 columns=ds.target_cols).assign(**{ds.pid: test_pid_s})
                                    else:
                                        mdl = mdl_cls(**mdl_kwargs)
                                        # train and predict
                                        mdl.fit(train_data.loc[:, input_feats], train_data.loc[:, ds.target_cols])
                                        log["end_time"] = pd.Timestamp.now().round(freq='S')
                                        train_pred = pd.DataFrame(target_scaler.inverse_transform(mdl.predict(train_data.loc[:, input_feats])),
                                                                  columns=ds.target_cols).assign(**{ds.pid: train_pid_s})
                                        test_pred = pd.DataFrame(target_scaler.inverse_transform(mdl.predict(test_data.loc[:, input_feats])),
                                                                 columns=ds.target_cols).assign(**{ds.pid: test_pid_s})
                                    mdl_size = get_model_size(mdl, mdl_lbl)

                                if rep > 0:
                                    # unshuffle
                                    train_data_gtruth = train_data.loc[:, ds.target_cols]\
                                        .assign(orig_idx=orig_train_row_idx)\
                                        .sort_values(by='orig_idx', ascending=True)\
                                        .drop(columns=['orig_idx']).reset_index(drop=True)
                                    if train_pred is not None:
                                        train_pred = train_pred.assign(orig_idx=orig_train_row_idx)\
                                            .sort_values(by='orig_idx', ascending=True)\
                                            .drop(columns=['orig_idx']).reset_index(drop=True)
                                else:
                                    train_data_gtruth = train_data.loc[:, ds.target_cols]

                                # score
                                if train_pred is not None:
                                    train_score = mse(target_scaler.inverse_transform(
                                        train_data_gtruth), train_pred.loc[:, ds.target_cols])
                                else:
                                    train_score = np.NaN
                                test_score = mse(target_scaler.inverse_transform(
                                    test_data.loc[:, ds.target_cols]), test_pred.loc[:, ds.target_cols])

                                print(f"{mdl_lbl:8} | fe: {fe_scheme:10} | has_ewma/s: {has_ewmas:1}/{has_ewmsts:1} | norm: {norm_scheme_lbl:10} | "
                                      f"seed: {rep} | train mse: {train_score:5.2f} K² | test mse: {test_score:5.2f} K²")

                                # track intermediate result
                                log["model_size"] = mdl_size
                                # trajectories need to be normalized for the internal workings of DBM
                                log["all_predictions_df"] = (test_pred / ds.temperature_scale)\
                                    .assign(**{'repetition': rep, ds.pid: test_pid_s})
                                log["ground_truth"] = (ds.data.loc[:, ds.target_cols + [ds.pid]]
                                                         .query(f"{ds.pid} in @test_fold_l")
                                                         .reset_index(drop=True) / ds.temperature_scale)\
                                    .assign(**{ds.pid: test_pid_s})\

                                log["seed"] = rep
                                return log

                        # conduct trainings and evaluations
                        if mdl_lbl == 'mlp':
                            with Parallel(n_jobs=-1) as prll:
                                experiment_logs = prll(delayed(run_static_training)(i) for i in range(n_seeds))
                        else:
                            experiment_logs = []
                            for rep in range(n_seeds if mdl_lbl in ("et", "rf", "mlp", "histgbm", 'lasso', 'svr') else 1):
                                experiment_logs.append(run_static_training(rep))

                        # generate tag
                        run_tag = run_tag or f"fe_{fe_scheme} norm_{norm_scheme_lbl} ewm adjust={EWM_ADJUST}"

                        # save to DB
                        ds.input_cols = input_feats  # updating dataset is important for DBM.save()
                        dbm = DBManager(model_tag=mdl_lbl, loss_metric='mse', cv=cv_lbl, scriptname=pathlib.Path(sys.argv[0]).name,
                                        hostname=platform.uname().node, debugmode=debug, dataset=ds, n_folds=1,
                                        model_size=experiment_logs[0]["model_size"], comment=run_tag,
                                        model_arch=experiment_logs[0].get('models_arch', str(
                                            mdl_kwargs)),  # mdl_kwargs won't include random_state
                                        save_model=mdl_lbl == 'mlp', save_trends=mdl_lbl == 'mlp')
                        dbm.ESTIMATES_DUMP_PATH = DUMP_PATH
                        dbm.save(experiment_logs)
    print("done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate static black box models')
    parser.add_argument('-t', '--tag', default="", required=False,
                        help='an identifier/tag/comment for the trials')
    args = parser.parse_args()
    run_blackbox_static(args.tag)
