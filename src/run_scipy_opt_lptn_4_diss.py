import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy
import scipy.io as sio
from tqdm import tqdm
import warnings
from pprint import pprint
from joblib import Parallel, delayed
from mawkutils.data import KaggleDataSet, DBManager as DBM
from mawkutils.topology import WallscheidLPTN
from hpo_lptn_4_diss import param_space as param_bounds

DATA_PATH = Path().cwd().parent / "data" / "input"
DATA_OUT_PATH = DATA_PATH.parent / 'output' / 'misc'

ds = KaggleDataSet(with_extra_cols=True)
ds.featurize(scheme='basic')
# add interpolated ploss and schlepp

# load up LUTs
ploss_tup = sio.loadmat(DATA_PATH / 'Losses_250V.mat')['Losses'][0][0]
ploss_lut = pd.DataFrame(ploss_tup[2], index=ploss_tup[0].ravel(),
                         columns=ploss_tup[1].ravel())
schlepp_tup = sio.loadmat(DATA_PATH / 'Schleppkennlinie.mat')['Schlepp'][0][0]
schlepp_arr = np.array([schlepp_tup[0], schlepp_tup[1]]).squeeze()
schlepp_lut = pd.Series(schlepp_arr[1], index=schlepp_arr[0])

# add LUT info to dataframe
codes, uniques = ploss_lut.index.factorize(sort=False)
idx_map = {k:v for k, v in zip(uniques, codes)}
codes, uniques = ploss_lut.columns.factorize(sort=False)
col_map = {k:v for k, v in zip(uniques, codes)}
ds.data.loc[:, 'iron_loss'] =\
     ploss_lut.to_numpy()[(np.sign(ds.data.torque.to_numpy()) * (np.round(ds.data.i_s / np.sqrt(2) / 2.5) * 2.5)).map(idx_map),
                    (np.round(ds.data.motor_speed.abs() / 100) * 100).map(col_map)]

ds.data['schlepp'] = schlepp_lut.loc[
    (np.round(ds.data.motor_speed.abs() / 100) * 100).tolist()].to_numpy()

hpo_stats = pd.read_pickle(DATA_OUT_PATH / 'custom_hpo_lptn.pkl.zip')
genset = [16, 20, 48, 53, 60]

def evaluate_coeffs_on_certain_profiles(coeffs_arr, p_set):
    mse_l = []
    l_infty_l = []
    mdl = WallscheidLPTN(ds.target_cols, coeffs_list=coeffs_arr)
    for pid, df in ds.data.query(f"{ds.pid} in @p_set").groupby(ds.pid):
        y_hat = np.clip(mdl.solve_ivp(df.reset_index(drop=True)), -100, 300)  # arbitrary
        gtruth = df.loc[:, mdl.target_cols].to_numpy()
        mse = np.mean((y_hat-gtruth)**2)
        l_infty = np.max(np.abs(y_hat - gtruth))
        mse_l.append(mse)
        l_infty_l.append(l_infty)
    return np.mean(mse_l), np.max(np.array(l_infty_l))



top5 = hpo_stats.dropna(axis=0).query("values < 20").sort_values("values", ascending=True).iloc[:5, :]
training_set = [p for p in ds.data.profile_id.unique() if p not in genset + [65]]


def evaluate_on_training_set(coeffs):
    return evaluate_coeffs_on_certain_profiles(coeffs, training_set)[0]



    
def scipy_opt_LPTN(ser, top_i):
    x0_cost = ser.loc["values"]
    pbar = tqdm(desc="Scipy minimize", position=top_i)
    def pbar_callback(xk, *args):
        pbar.update(1)
        cost = evaluate_on_training_set(xk)
        pbar.set_postfix_str(f"MSE: {cost:.2f} A², diff to u0: {cost - x0_cost:.2f} A²")
        pbar.refresh()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = scipy.optimize.minimize(fun=evaluate_on_training_set, x0=ser.iloc[:-1], 
                    bounds=list(param_bounds.values()),
                    callback=pbar_callback,
                    method="SLSQP", options=dict(ftol=1e-8, eps=1e-3, max_iter=100, disp=True),
                        #"SLSQP",  
                        #"TNC", 
                    tol=1e-8)
    print(res.message)
    print(f"Particle {top_i}: Found optimum after {res.nit} at: {res.fun}")
    return res.x

with Parallel(n_jobs=len(top5)) as prll:
    ret = prll(delayed(scipy_opt_LPTN)(row, i) for i, row in top5.iterrows())

pprint(ret)
    