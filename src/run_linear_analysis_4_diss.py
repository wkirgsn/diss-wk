"""linear modeling experiments for black box chapter in WK dissertation, 2022"""
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from run_blackbox_static_4_diss import run_blackbox_static, get_model_size, CustomScaler, add_features, normalize_data
from mawkutils.data import KaggleDataSet, SmoothKaggleDataSet, DBManager, add_mov_stats

DEBUG = False
CV = 'hpo_1fold_no_val_static_diss'
DUMP_PATH = pathlib.Path.cwd().parent / 'data' / 'output' / 'static_blackbox_estimates'
DUMP_PATH.mkdir(exist_ok=True, parents=True)
N_SEEDS = 3 if DEBUG else 10
RUN_TAG = 'lin mdl analysis'
FE_SCHEME = ['extensive']
EWMA_FLAGS = [True]
EWMS_FLAGS = [False]
SCALERS_D = {'custom': CustomScaler}
OPT_SPANS = [10290, 699, 13570]


def rand_search_ewmas():
    "Random search over an increasing number of span values/EWMAs with OLS"
    def get_spreaded_random_integers(size, seed=0):
        low, high = 120, 14401
        domain = np.arange(low, high)
        rng = np.random.default_rng(seed=seed)
        sampled = []
        for _ in range(size):
            for sample in sampled:
                domain = domain[(domain < (sample-600)) | (domain > (sample + 600))]
            sampled.append(rng.choice(domain))
        return sampled

    with Parallel(n_jobs=11) as prll:
        for n_spans in range(1, 11):
            ret = prll(delayed(run_blackbox_static)(
                run_tag=RUN_TAG+' rand_search_ewmas', fe_schemes=FE_SCHEME,
                ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS,
                zoo_cfg_d=dict(ridge=(Ridge, {}), ols=(LinearRegression, {})),
                spans=get_spreaded_random_integers(size=n_spans, seed=rep),
                scalers_d=SCALERS_D, debug=DEBUG, cv=CV)
                for rep in trange(33, desc=f"n_spans={n_spans}:"))


def grid_search_reg_lasso_ridge():
    "Grid search over regularization alpha in Lasso and Ridge"
    alpha_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    opt_spans = OPT_SPANS
    with Parallel(n_jobs=11 if not DEBUG else 1) as prll:
        ret = prll(delayed(run_blackbox_static)(
            run_tag=RUN_TAG+' grid_search_reg_lasso_ridge', fe_schemes=FE_SCHEME,
            ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS,
            zoo_cfg_d=dict(ridge=(Ridge, dict(alpha=a)), lasso=(Lasso, dict(alpha=a))),
            spans=opt_spans, scalers_d=SCALERS_D, debug=DEBUG, cv=CV, n_seeds=9) for a in alpha_range)


def grid_search_trainset_size():
    "Grid search over increasing (per-profile) shuffled training set size with OLS"

    black_listed_profiles = [16, 20, 48, 53, 60]  # gen set
    test_profiles = [65]

    ds = SmoothKaggleDataSet()
    train_l, _, _ = ds.get_profiles_for_cv('hpo_1fold_no_val_static_diss')
    pid_sizes = ds.get_pid_sizes()
    train_size_total = pid_sizes.loc[train_l[0]].sum()  # 100% = 1157138 samples
    grp_ds = ds.data.groupby(ds.pid, sort=False)

    def get_reduced_ds():
        for d in range(2, 10):
            if d==5: continue
            train_size = int(train_size_total * d * 0.1)
            # define repeats
            repeats = 1 if d == 1 else 50
            for r in range(repeats):
                dfs_l = [df for p, df in grp_ds if p not in test_profiles+black_listed_profiles]
                random.shuffle(dfs_l)
                new_train_df = pd.concat(dfs_l, ignore_index=True).iloc[:train_size, :]

                ds_copy = SmoothKaggleDataSet()
                ds_copy.data = pd.concat([new_train_df] + [grp_ds.get_group(tst_p) for tst_p in test_profiles],
                                         ignore_index=True)
                yield ds_copy, d, r

    with Parallel(n_jobs=-1 if not DEBUG else 1) as prll:

        ret = prll(delayed(run_blackbox_static)(
            run_tag=RUN_TAG+f' rand_search_trainset_size {d_=} {r_=}', fe_schemes=FE_SCHEME,
            ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS,
            zoo_cfg_d=dict(ols=(LinearRegression, {}),
                           ridge=(Ridge, dict(alpha=0.5)), lasso=(Lasso, dict(alpha=4e-5))),
            spans=OPT_SPANS, scalers_d=SCALERS_D, debug=DEBUG, cv=CV, n_seeds=1, ds=ds_)
            for ds_, d_, r_ in get_reduced_ds())


def VIF_feat_sel():
    "feature elimination according to largest VIF"

    VIF_FACTORS_FILE = DUMP_PATH / "vif_factors.csv"

    with Parallel(n_jobs=-1 if not DEBUG else 1) as prll:
        if pathlib.Path.exists(VIF_FACTORS_FILE):
            vifs_df = pd.read_csv(VIF_FACTORS_FILE)
        else:
            # create data set
            ds = SmoothKaggleDataSet()
            if DEBUG:
                ds.data = pd.concat([df.iloc[:1000, :] for p, df
                                    in ds.data.groupby(ds.pid, sort=False)], ignore_index=True)  # crop
            # extensive FE additions
            proc_data = add_features(ds.data, 'extensive')
            input_feats = [c for c in proc_data if c not in ds.target_cols + [ds.pid]]
            proc_data = add_mov_stats(proc_data, input_feats, ds.pid, OPT_SPANS, add_ewma=True, add_ewms=False)
            input_feats = [c for c in proc_data if c not in ds.target_cols + [ds.pid]]
            # rearrange
            proc_data = proc_data.loc[:, input_feats + [ds.pid] + ds.target_cols]
            # normalize
            proc_data, input_scaler, target_scaler = normalize_data(
                proc_data, input_feats, ds.target_cols, SCALERS_D['custom'])
            # CV
            train_l, val_l, test_l = ds.get_profiles_for_cv("hpo_1fold_no_val_static_diss", verbose=False)

            train_data = proc_data.loc[proc_data.loc[:, ds.pid].isin(train_l[0]), input_feats].reset_index(drop=True)
            reduced_data = train_data.copy()
            vifs_l = []
            while reduced_data.shape[1] > 4:
                vifs = prll(delayed(variance_inflation_factor)(reduced_data, i) for i in range(reduced_data.shape[1]))
                vifs_ser = pd.Series(vifs, index=reduced_data.columns.tolist())
                vifs_l.append(vifs_ser)
                # drop highest vif(s) columns
                reduced_data = reduced_data.loc[:, vifs_ser.loc[vifs_ser != vifs_ser.max()].index.tolist()]
            # save calculated vifs
            vifs_df = pd.concat(vifs_l, axis=1).T
            vifs_df.to_csv(VIF_FACTORS_FILE, index=False)
        # compute performance for the different reduced input feature sets
        ret = prll(delayed(run_blackbox_static)(run_tag=RUN_TAG+' vif_feat_sel', fe_schemes=FE_SCHEME,
                                                ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS,
                                                zoo_cfg_d=dict(ols=(LinearRegression, {}), ridge=(
                                                    Ridge, dict(alpha=0.5)), lasso=(Lasso, dict(alpha=4e-5))),
                                                spans=OPT_SPANS, scalers_d=SCALERS_D, debug=DEBUG, cv=CV, n_seeds=1,
                                                custom_input_feature_whitelist=row.loc[~row.isna()].index.tolist()) for _, row in vifs_df.iterrows())


def RFE_feat_sel():
    "recursive feature elimination (drop least valued linear coefficients)"
    RFE_FACTORS_FILE = DUMP_PATH / "rfe_lasso_coeffs.csv"

    with Parallel(n_jobs=-1 if not DEBUG else 1) as prll:
        if pathlib.Path.exists(RFE_FACTORS_FILE):
            rfes_df = pd.read_csv(RFE_FACTORS_FILE)
        else:
            # create data set
            ds = SmoothKaggleDataSet()
            if DEBUG:
                ds.data = pd.concat([df.iloc[:1000, :] for p, df
                                    in ds.data.groupby(ds.pid, sort=False)], ignore_index=True)  # crop
            # extensive FE additions
            proc_data = add_features(ds.data, 'extensive')
            input_feats = [c for c in proc_data if c not in ds.target_cols + [ds.pid]]
            proc_data = add_mov_stats(proc_data, input_feats, ds.pid, OPT_SPANS, add_ewma=True, add_ewms=False)
            input_feats = [c for c in proc_data if c not in ds.target_cols + [ds.pid]]
            # rearrange
            proc_data = proc_data.loc[:, input_feats + [ds.pid] + ds.target_cols]
            # normalize
            proc_data, input_scaler, target_scaler = normalize_data(
                proc_data, input_feats, ds.target_cols, SCALERS_D['custom'])
            # CV
            train_l, val_l, test_l = ds.get_profiles_for_cv("hpo_1fold_no_val_static_diss", verbose=False)

            train_data = proc_data.loc[proc_data.loc[:, ds.pid].isin(train_l[0]), :].reset_index(drop=True)
            reduced_data = train_data.copy()
            rfes_l = []
            while len(input_feats) > 4:
                mdl = Lasso(alpha=4e-5)
                mdl.fit(reduced_data.loc[:, input_feats], reduced_data.loc[:, ds.target_cols])
                coeffs = pd.Series(np.abs(mdl.coef_).mean(axis=0), index=input_feats)
                rfes_l.append(coeffs)
                # drop highest rfe(s) columns
                input_feats = coeffs.loc[coeffs != coeffs.min()].index.tolist()
            # save calculated rfes
            rfes_df = pd.concat(rfes_l, axis=1).T
            rfes_df.to_csv(RFE_FACTORS_FILE, index=False)
        # compute performance for the different reduced input feature sets
        ret = prll(delayed(run_blackbox_static)(run_tag=RUN_TAG+' rfe_feat_sel', fe_schemes=FE_SCHEME,
                                                ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS,
                                                zoo_cfg_d=dict(ols=(LinearRegression, {}), ridge=(
                                                    Ridge, dict(alpha=0.5)), lasso=(Lasso, dict(alpha=4e-5))),
                                                spans=OPT_SPANS, scalers_d=SCALERS_D, debug=DEBUG, cv=CV, n_seeds=1,
                                                custom_input_feature_whitelist=row.loc[~row.isna()].index.tolist()) for _, row in rfes_df.iterrows())


AGENDA = [  # rand_search_ewmas,
    # grid_search_reg_lasso_ridge,
    grid_search_trainset_size,
    # VIF_feat_sel,
    # RFE_feat_sel,
]


if __name__ == '__main__':
    """Repeat Analysis from ISIE 2019 paper"""

    for func in AGENDA:
        func()
