import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

from mawkutils.data import KaggleDataSet, ChunkedKaggleDataSet, DBManager as DBM
from mawkutils.topology import WallscheidLPTN
from mawkutils.experiments import evaluate_timeseries_nn
from hpo_lptn_4_diss import param_space

DATA_IN_PATH = Path().cwd().parent / "data" / "input"
DATA_OUT_PATH = DATA_IN_PATH.parent / 'output' / 'misc'
Path.mkdir(DATA_OUT_PATH, parents=True, exist_ok=True)

CV = '1fold_static_diss'
N_TRIALS = 10_000
TORCH_LEARN = True
CONTINUE_WITH_2ND_ORDER_OPT = False

def main():
    data_cls = ChunkedKaggleDataSet if TORCH_LEARN else KaggleDataSet
    extra_d = dict(chunk_size=5870, cv=CV) if TORCH_LEARN else {}
    ds = data_cls(with_extra_cols=True, **extra_d)
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
    idx_map = {k:v for k, v in zip(uniques, codes)}
    codes, uniques = ploss_lut.columns.factorize(sort=False)
    col_map = {k:v for k, v in zip(uniques, codes)}
    
    ds.data.loc[:, 'iron_loss'] =\
        ploss_lut.to_numpy()[(np.sign(ds.data.torque.to_numpy()) * (np.round(ds.data.i_s / np.sqrt(2) / 2.5) * 2.5)).map(idx_map),
                        (np.round(ds.data.motor_speed.abs() / 100) * 100).map(col_map)]

    ds.data['schlepp'] = schlepp_lut.loc[(np.round(ds.data.motor_speed.abs() / 100) * 100).tolist()].to_numpy()

    if TORCH_LEARN:

        ds.input_cols += ['iron_loss', 'schlepp']
        ds.data = ds.data.loc[:, ds.input_cols + [ds.pid] + ds.target_cols]  # order is important due to generate_tensor()
        ds.normalize()
        if CONTINUE_WITH_2ND_ORDER_OPT:
            # read out best seeds and continue optimizing from there
            expid = 85251  # chosen from inspecting DB
            meta_info_lptn = DBM.query(f"SELECT* FROM trials as t LEFT JOIN experiments as e ON t.experiment_id = e.id "
                                        f"where experiment_id = {expid} "
                                        f"ORDER BY t.mse ASC "
                                        "LIMIT 5;")
            def run_experiment(s):
                n_epochs = 160
                load_model_path = DBM.MODEL_DUMP_PATH / f"exp_{expid}_seed_{s}_fold_0.pt"
                exp_logs = evaluate_timeseries_nn(run_tag=f"lbfgs TorchLPTN, seed {s}, expid {expid}", model_tag='lptn', n_batches=4, ds=ds, tbptt_size=720, # 100 min
                                n_epochs=n_epochs, cv_method=CV, opt_func=torch.optim.LBFGS, lr=5e-2, n_jobs=1,
                                n_seeds=1, half_lr_at=[80, 120], continue_mdl=load_model_path, opt_args=dict(max_iter=2, history_size=10))
            with Parallel(n_jobs=len(meta_info_lptn)) as prll:
                ret = prll(delayed(run_experiment)(s) for s in meta_info_lptn.seed.tolist())
        else:
            exp_logs = evaluate_timeseries_nn(run_tag="run TorchLPTN", model_tag="lptn", n_batches=4, ds=ds, tbptt_size=558, # 10 min
                                n_epochs=160, cv_method=CV, opt_func=torch.optim.NAdam, lr=20e-3, n_jobs=8,
                                n_seeds=16, half_lr_at=[80, 120],)
        #pd.DataFrame([l["models_state_dict"] for l in exp_logs])  # save coeffs somehow?
    else:
        # random search with numpy/numba

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

        def sample_n_evaluate(mdl):
            coeffs = []
            
            for k, space in param_space.items():
                # ignore logs for now
                coeffs.append(np.random.rand(1).item() * (space[1] - space[0]) + space[0])
            mdl.coeffs_list = tuple(coeffs)
            score = evaluate_all_profiles(mdl)
            return coeffs + [score]
        
        with Parallel(n_jobs=-1) as prll:
            score_sheet_l = prll(delayed(sample_n_evaluate)(mdl) for _ in tqdm(range(N_TRIALS)))
        
        score_sheet_df = pd.DataFrame(score_sheet_l, columns=list(param_space.keys())+['MSE'])
        score_sheet_df.sort_values('MSE', ascending=True)

        score_sheet_df.to_csv(DATA_OUT_PATH / 'random_search.csv', index=False)

if '__main__' == __name__:
    main()
