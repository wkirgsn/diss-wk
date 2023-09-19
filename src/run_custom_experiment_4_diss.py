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
from lightgbm import LGBMRegressor
import torch
from torch import nn
from mawkutils.data import KaggleDataSet, SmoothKaggleDataSet, DBManager
from mawkutils.topology import MLP
from run_blackbox_static_4_diss import CustomScaler, run_blackbox_static


DEBUG = False
CV = '1fold_no_val_static_diss'#'hpo_1fold_no_val_static_diss'
DUMP_PATH = pathlib.Path.cwd().parent / 'data' / 'output' / 'static_blackbox_estimates'
DUMP_PATH.mkdir(exist_ok=True, parents=True)
N_SEEDS = 3 if DEBUG else 10
BATCH_SIZE = 2**8  # only for MLP (2**11)
N_EPOCHS = 100  # only for MLP (400)
EWM_ADJUST = False

FE_SCHEMES = [#'plain', 'basic',
 'extensive']
EWMA_FLAGS = [True]
EWMS_FLAGS = [False]
OPT_SPANS = [10290, 699, 13570]
SPANS = [1320, 3360, 6360, 9480]
BEST_INP_COLS = ['coolant',
 'u_d',
 'u_q',
 'motor_speed',
 'i_d',
 'i_q',
 'i_s',
 'u_s',
 'S_el',
 'P_el',
 'i_s_x_w',
 'S_x_w',
 'coolant_ewma_10290',
 'u_d_ewma_10290',
 'u_q_ewma_10290',
 'i_q_ewma_10290',
 'i_s_ewma_10290',
 'S_el_ewma_10290',
 'S_x_w_ewma_10290',
 'coolant_ewma_699',
 'u_d_ewma_699',
 'u_q_ewma_699',
 'i_d_ewma_699',
 'i_q_ewma_699',
 'i_s_ewma_699',
 'u_s_ewma_699',
 'S_el_ewma_699',
 'P_el_ewma_699',
 'S_x_w_ewma_699',
 'ambient_ewma_13570',
 'coolant_ewma_13570',
 'u_d_ewma_13570',
 'u_q_ewma_13570',
 'i_d_ewma_13570',
 'i_q_ewma_13570',
 'i_s_ewma_13570',
 'u_s_ewma_13570',
 'P_el_ewma_13570',
 'i_s_x_w_ewma_13570']

ZOO_d = {
    #'ols': (LinearRegression, {}),
    #"ridge": (Ridge, {'alpha': 5e-1}),
    "lasso": (Lasso, {'alpha': 4e-5}),
    # 'svr': (LinearSVR, {'dual': False, 'loss': 'squared_epsilon_insensitive'}),
    # 'rf': (RandomForestRegressor, {"n_jobs": -1, "max_depth": 10, "n_estimators": 20}),
    # 'et': (ExtraTreesRegressor,  {"n_jobs": -1, "max_depth": 10, "n_estimators": 20}),
    # 'lgbm': (LGBMRegressor, {"n_jobs": -1, "max_depth": 5, "n_estimators": 50}),
    #'histgbm': (HistGradientBoostingRegressor, {"max_depth": 5, "max_iter": 50}),
    # 'mlp': (MLP, {})

}


SCALERS = {
    #'standard': StandardScaler,# 'minmax': MinMaxScaler,
    'custom': CustomScaler
 }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate static black box models')
    parser.add_argument('-t', '--tag', default="", required=False,
                        help='an identifier/tag/comment for the trials')
    args = parser.parse_args()
    run_blackbox_static(args.tag, fe_schemes=FE_SCHEMES, ewma_flags=EWMA_FLAGS, ewms_flags=EWMS_FLAGS, zoo_cfg_d=ZOO_d,
            spans=OPT_SPANS, scalers_d=SCALERS, cv=CV, debug=DEBUG, n_seeds=1,
            #custom_input_feature_whitelist=BEST_INP_COLS,
            )
