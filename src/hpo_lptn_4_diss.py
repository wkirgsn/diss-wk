import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import platform
from numba import njit
import optuna
import scipy.io as sio
import os
from joblib import Parallel, delayed
import sshtunnel
import polars as pl

from optuna.samplers._tpe.sampler import default_gamma, default_weights
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters

from mawkutils.data import KaggleDataSet, DBManager as DBM
from mawkutils.topology import WallscheidLPTN

DATA_IN_PATH = Path().cwd().parent / "data" / "input"
DATA_OUT_PATH = DATA_IN_PATH.parent / 'output' / 'misc'
Path.mkdir(DATA_OUT_PATH, parents=True, exist_ok=True)

N_TRIALS = 10_000
BATCH_SIZE = 120
USE_WALLSCHEID_AS_GBEST = False

param_space = {'caps0': (1e2, 1e5),  # log
               'caps1': (1e2, 1e5),  # log
               'caps2': (1e2, 1e5,),  # log
               'caps3': (1e2, 1e5,),  # log
               'const_Rs_sy_sw': (1e-2, 1, ),  # log
               'const_Rs_sy_st': (1e-2, 1, ),  # log
               'const_Rs_sw_st': (1e-2, 1, ),  # log
               'lin_Rs_slope': (-1e-2, -1e-4, ), # log
               'lin_Rs_bias': (1e-4, 1e-1, ),  # log
               'exp_Rs_magn0': (1e-1, 4, ),
               'exp_Rs_magn1': (1e-1, 4, ),
               'exp_Rs_magn2': (1e-1, 4, ),
               'exp_Rs_b0': (1e-2, 50e-2, ),
               'exp_Rs_b1': (1e-2, 50e-2, ),
               'exp_Rs_b2': (1e-2, 50e-2, ),
               'exp_Rs_a0': (1e-2, 1, ),
               'exp_Rs_a1': (1e-2, 1, ),
               'exp_Rs_a2': (1e-2, 1, ),
               'bipoly_Rs_magn': (1e-4, 1, ),  # log
               'bipoly_Rs_a': (-1, 0, ),
               'bipoly_Rs_b': (0, 1, ),
               'bipoly_Rs_c': (0, 1, ),
               'ploss_Rdc': (1e-4, 1, ),  # log
               'ploss_alpha_cu': (1e-5, 1e-2, ), #log
               'ploss_alpha_ac_1': (0.1, 1, ),
               'ploss_alpha_ac_2': (0.1, 1, ),
               'ploss_beta_cu': (0.1, 5, ),
               'ploss_k_1_0': (0, 1, ),
               'ploss_k_1_1': (0, 1, ),
               'ploss_k_1_2': (0, 1, ),
               'ploss_k_1_3': (-1, 1, ),
               'ploss_k_2': (0, 1, ),
               'ploss_alpha_fe': (-1e-2, 0, ),
               'schlepp_factor': (0.1, 10, )}

log_spaces_l = [k for k in param_space if k.startswith(("caps", "const_Rs_", "lin_Rs_bias", "exp_Rs_magn",
                                                        "exp_Rs_a", "bipoly_Rs_magn", "ploss_Rdc", "ploss_alpha_cu"))]
log_idx = [i for i, k in enumerate(param_space.keys()) if k in log_spaces_l]

class PSO:
    def __init__(self, pop_size=120, w=0.9, c1=2.0, c2=2.0):
        self.epoch = 0
        self.w = w
        self.c1 = c1
        self.c2 = c2
        param_space_df = pd.DataFrame(param_space)
        param_space_df.loc[:, log_spaces_l] = np.log10(param_space_df.loc[:, log_spaces_l])
        self.low, self.high = param_space_df.iloc[0].to_numpy().ravel(), param_space_df.iloc[1].to_numpy().ravel()
        self.pbest_conf = np.random.rand(pop_size, len(param_space)) * (self.high -
                                                                        self.low).reshape(1, -1) + self.low.reshape(1, -1)
        self.pbest_values = np.full((pop_size,), np.inf)
        if USE_WALLSCHEID_AS_GBEST:
            self.gbest_conf = np.array([getattr(WallscheidLPTN, a) for a in param_space])
            self.gbest_conf[log_idx] = np.log10(self.gbest_conf[log_idx])
            self.gbest_value = WallscheidLPTN.train_data_mse
        else:
            self.gbest_conf = self.pbest_conf.copy()
            self.gbest_value = np.inf
        self.X = self.pbest_conf.copy()
        self.V = np.random.randn(pop_size, self.pbest_conf.shape[1])
        self.values_log = []
        self.params_log = []

    def ask(self):
        if self.epoch > 0:
            pso_equation(self.pbest_conf, self.gbest_conf,
                         -(self.high - self.low)*0.1, (self.high - self.low)*0.1, self.low, self.high,
                         self.w, self.c1, self.c2, self.X, self.V)
        self.epoch += 1
        return self.X.copy()

    def tell(self, values):
        self.values_log.append(values.ravel())
        self.params_log.append(self.X)
        self.pbest_conf = np.where((values < self.pbest_values).reshape(-1, 1), self.X, self.pbest_conf)
        self.pbest_values = np.minimum(values, self.pbest_values)
        min_val_idx = np.argmin(np.nan_to_num(values, nan=np.inf))
        if values[min_val_idx] < self.gbest_value:
            self.gbest_conf = self.X[min_val_idx].reshape(1, -1)
            self.gbest_value = values[min_val_idx]


@njit
def pso_equation(pbest, gbest, low_v_bounds, high_v_bounds, low_bounds, high_bounds, w, c1, c2, X, V):
    """Source: https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/soo/nonconvex/pso.py"""
    n_particles, n_var = X.shape

    r1 = np.random.random((n_particles, n_var))
    r2 = np.random.random((n_particles, n_var))

    inertia = w * V
    cognitive = c1 * r1 * (pbest - X)
    social = c2 * r2 * (gbest - X)
    # calculate the velocity vector
    V[:] = np.minimum(np.maximum(inertia + cognitive + social, low_v_bounds), high_v_bounds)

    # displace particles
    X[:] = np.minimum(np.maximum(X + V, low_bounds), high_bounds)  # clip to bounds


class Study:

    param_space = param_space

    def __init__(self, log, name, sampler, i=0):
        self.log = log
        self.name = name
        self.i = i
        self.sampler = sampler


def create_study(storage=None, study_name=None, direction="minimize", sampler=None, load_if_exists=True, n_iters=100):

    storage = storage or Path.cwd().parent / 'data' / 'output' / 'misc' / 'custom_hpo_lptn.pkl.zip'
    if load_if_exists:
        if storage.exists():
            study_mat = pd.read_pickle(storage).to_numpy()
            study_mat = np.vstack([study_mat, np.zeros((n_iters - len(study_mat), study_mat.shape[1]))])
    else:
        study_mat = np.zeros((n_iters, len(param_space)+1))

    return Study(log=study_mat, sampler=sampler, name=study_name)


if __name__ == '__main__':
    ds = KaggleDataSet(with_extra_cols=True)
    ds.featurize(scheme='basic')

    # load up LUTs
    ploss_tup = sio.loadmat(DATA_IN_PATH / 'Losses_250V.mat')['Losses'][0][0]
    ploss_lut = pd.DataFrame(ploss_tup[2], index=ploss_tup[0].ravel(),
                             columns=ploss_tup[1].ravel())
    schlepp_tup = sio.loadmat(DATA_IN_PATH / 'Schleppkennlinie.mat')['Schlepp'][0][0]
    schlepp_arr = np.array([schlepp_tup[0], schlepp_tup[1]]).squeeze()
    schlepp_lut = pd.Series(schlepp_arr[1], index=schlepp_arr[0])

    # add LUT info to dataframe
    codes, uniques = ploss_lut.index.factorize(sort=False)
    idx_map = {k: v for k, v in zip(uniques, codes)}
    codes, uniques = ploss_lut.columns.factorize(sort=False)
    col_map = {k: v for k, v in zip(uniques, codes)}
    ds.data.loc[:, 'iron_loss'] =\
        ploss_lut.to_numpy()[(np.sign(ds.data.torque.to_numpy()) * (np.round(ds.data.i_s / np.sqrt(2) / 2.5) * 2.5)).map(idx_map),
                             (np.round(ds.data.motor_speed.abs() / 100) * 100).map(col_map)]

    ds.data['schlepp'] = schlepp_lut.loc[
        (np.round(ds.data.motor_speed.abs() / 100) * 100).tolist()].to_numpy()

    train_l, _, _ = ds.get_profiles_for_cv("1fold_static_diss")  # exclude gen set and val set
    train_l = train_l[0]
    ds.data = ds.data.query(f"{ds.pid} in @train_l")  # filter for train set

    # solve all profiles
    def evaluate_all_profiles(mdl_):
        mse_l = []
        for pid, df in ds.data.groupby(ds.pid):
            y_hat = np.clip(mdl_.solve_ivp(df.reset_index(drop=True)), -100, 300)  # arbitrary
            gtruth = df.loc[:, mdl_.target_cols].to_numpy()
            mse_l.append(np.mean((y_hat-gtruth)**2))
        return np.mean(mse_l)

    # init model
    score_sheet_l = []
    mdl = WallscheidLPTN(target_cols=ds.target_cols)
    # JIT compile
    evaluate_all_profiles(mdl)

    # Optimization with Tree Parzen Estimator
    DB_NAME = 'optuna'
    STUDY_NAME = 'study_lptn_4_diss_TPE_1'
    SERVER_LOCAL_PORT2PSQL = 6432
    PC2_LOCAL_PORT2PSQL = 12000

    def evaluate(coeffs):
        mdl.coeffs_list = tuple(coeffs)
        score = evaluate_all_profiles(mdl)
        return score

    
    pbar = tqdm(range(N_TRIALS))
    pso = PSO(pop_size=BATCH_SIZE)
    with Parallel(n_jobs=-1) as prll:
        for trial_i in pbar:
            coeffs_mat = pso.ask()
            coeffs_mat[:, log_idx] = 10**(coeffs_mat[:, log_idx])
            coeffs_l = coeffs_mat.tolist()
            rets = prll(delayed(evaluate)(coeffs) for coeffs in coeffs_l)
            pso.tell(np.asarray(rets))
            pbar.set_postfix_str(f"Best:  {pso.gbest_value:.2f} K²")

    # store to disk
    print("Store to disk..")
    df = pd.DataFrame(np.vstack(pso.params_log), columns=[k for k in param_space])
    df.loc[:, 'values'] = np.hstack(pso.values_log)
    df.loc[:, log_spaces_l] = 10 ** df.loc[:, log_spaces_l]
    df.to_pickle(DATA_OUT_PATH / 'custom_hpo_lptn.pkl.zip')

    # print result
    print("Best is", )
    print(pso.gbest_conf)
    print("with value")
    print(f"{pso.gbest_value:.2f} K²")
