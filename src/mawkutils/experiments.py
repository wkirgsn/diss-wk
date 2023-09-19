from torchdiffeq import odeint_adjoint, odeint
import torch.nn as nn
import torch
from torchinfo import summary as ti_summary
import numpy as np
from tqdm import trange, tqdm
from joblib import delayed, Parallel
from pathlib import Path
import pandas as pd
import sys
import random
import platform
import json
import subprocess
from pprint import pprint

from mawkutils.topology import (AdjointConformTNN, DifferenceEqLayer,
                                TNNCell, ElSystemResponseCell, ExplEulerCell, CaplessTNNCell, SingleSubNNTNNCell,
                                ResidualLSTM, ResidualGRU, TemporalConvNet, LPTNTorch)
from mawkutils.data import DBManager, KaggleDataSet, add_mov_stats
from mawkutils.validation import generate_tensor, SampleWeightedMSELoss, calc_metrics

diff_eq_like_archs = ['tnn', 'capless_tnn', 'single_sub_tnn', 'expleuler', 'el_ssnn', 'lptn']
known_model_archs = diff_eq_like_archs + ['node', 'lstm', 'gru', 'tcn']


def get_initial_hidden_tensor(tens, n_targets, model_tag='tnn', layer_config=None):
    if model_tag in diff_eq_like_archs:
        hidden = tens[0, :, -n_targets:]
    elif model_tag in ('lstm', 'gru'):
        hidden = [torch.zeros((tens.shape[1], lay_spec['units']),
                              dtype=tens.dtype, device=tens.device)
                  for lay_spec in layer_config['f']]
        hidden[-1] = tens[0, :, -n_targets:]
        if model_tag == 'lstm':
            # (hx, cx)
            hidden = (hidden, [torch.zeros_like(h) for h in hidden])
    else:
        raise NotImplemented(model_tag)
    return hidden


def evaluate_timeseries_nn(run_tag, debug=False, model_tag='tnn', n_batches=1, ds=None, loss_func=None,
                           tbptt_size=32, n_epochs=300, opt_func=None, lr=1e-2, half_lr_at=None, opt_args=None,
                           n_jobs=10, n_seeds=10, cv_method='1fold', layer_cfg_d=None, continue_mdl=None):

    train_pid = ds.pid
    print("Find a mapping from")
    print(ds.input_cols)
    print('to')
    print(ds.target_cols)
    pid_sizes = ds.get_pid_sizes().to_dict()  # test and val set sizes
    n_targets = len(ds.target_cols)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    opt_args = opt_args or {}
    assert model_tag in known_model_archs,\
        f"model architecture '{model_tag}' unknown. Should be in {known_model_archs}"

    # Training parameters
    epochs = 3 if debug else n_epochs
    train_l, val_l, test_l = ds.get_profiles_for_cv(cv_method)
    if debug:
        train_l = train_l[:8]
    opt_func = opt_func or torch.optim.NAdam
    loss_func = loss_func or SampleWeightedMSELoss

    def run_dyn_training(rep=0):
        # seed
        np.random.seed(rep)
        random.seed(rep)
        torch.manual_seed(rep)

        logs = {'loss_trends_train': [[] for _ in train_l],
                'loss_trends_val': [[] for _ in val_l],
                'models_state_dict': [],
                'start_time': pd.Timestamp.now().round(freq='S')
                }

        all_predictions_df = ds.data.loc[:, ds.target_cols + [ds.pid]]
        gtruth = all_predictions_df.copy()

        # iterate over k folds
        for fold_i, (fold_train_profiles, fold_val_profiles, fold_test_profiles) \
                in enumerate(zip(train_l, val_l, test_l)):
            fold_val_profiles = fold_val_profiles or []

            # generate tensors
            train_tensor, train_sample_weights = generate_tensor(fold_train_profiles, ds, device,
                                                                 pid_lbl=train_pid)
            val_tensor, val_sample_weights = generate_tensor(fold_val_profiles or [], ds, device)
            test_tensor, test_sample_weights = generate_tensor(fold_test_profiles, ds, device)

            # model init
            if model_tag == 'node':
                mdl = AdjointConformTNN(train_tensor[:, :, :len(ds.input_cols)],
                                        n_targets, ds.input_cols,
                                        temperature_cols=ds.temperature_cols,
                                        non_temperature_cols=ds.non_temperature_cols).to(device)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
            elif model_tag in ('tnn', 'capless_tnn', 'single_sub_tnn'):
                mdl_cls = {'tnn': TNNCell, 'capless_tnn': CaplessTNNCell, 'single_sub_tnn': SingleSubNNTNNCell}[model_tag]
                mdl = DifferenceEqLayer(mdl_cls, n_targets=n_targets, input_feats=ds.input_cols, temperature_cols=ds.temperature_cols,
                                        sample_time=ds.sample_time, non_temperature_cols=ds.non_temperature_cols, layer_cfg=layer_cfg_d).to(device)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
            elif model_tag == 'expleuler':
                mdl = DifferenceEqLayer(ExplEulerCell, n_targets=n_targets, input_feats=ds.input_cols, temperature_cols=ds.temperature_cols,
                                        sample_time=ds.sample_time, layer_cfg=layer_cfg_d).to(device)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
            elif model_tag == 'el_ssnn':
                mdl = DifferenceEqLayer(ElSystemResponseCell, input_cols=ds.input_cols, target_cols=ds.target_cols,
                                        sample_time=ds.sample_time,).to(device)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
            elif model_tag in ['lstm', 'gru', 'tcn']:
                layer_cls = {'lstm': ResidualLSTM, 'gru': ResidualGRU, 'tcn': TemporalConvNet}[model_tag]
                mdl = layer_cls(len(ds.input_cols), layer_cfg=layer_cfg_d)
            elif model_tag == 'lptn':
                # Wallscheid LPTN style with free constants
                mdl = DifferenceEqLayer(LPTNTorch, input_cols=ds.input_cols, target_cols=ds.target_cols,
                                         temp_scale=ds.temperature_scale)
                mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2

            if continue_mdl is not None:
                assert isinstance(continue_mdl, (str, Path)), "continue_mdl should be a path to a model state dict"
                mdl.load_state_dict(torch.load(continue_mdl))

            opt = opt_func(mdl.parameters(), lr=lr, **opt_args)
            loss = loss_func().to(device)

            pbar = trange(epochs, desc=f"Seed {rep}, fold {fold_i}", position=rep*len(train_l)+fold_i, unit="epoch")

            if rep == 0 and fold_i == 0:  # print only once
                # data is usually (#TimeSeriesLength, #BatchSize, #Features)
                if model_tag == 'node':
                    info_kwargs = {"x": train_tensor[0, :, -n_targets:],
                                   "t": t_span_train[0]}
                    mdl_info = ti_summary(mdl, input_data=info_kwargs, device=device, verbose=0)
                elif model_tag in ['lstm', 'gru'] + diff_eq_like_archs:
                    info_kwargs = {}
                    if model_tag in diff_eq_like_archs:
                        info_kwargs['state'] = torch.randn((1, n_targets),
                                                           dtype=torch.float, device=device)
                    elif model_tag in ['gru', 'lstm']:
                        info_kwargs['state'] = [torch.randn((1, l_spec['units']), device=device) for
                                                l_spec in layer_cfg_d['f']]
                        if model_tag == 'lstm':
                            info_kwargs['state'] = (info_kwargs['state'], info_kwargs['state'])
                    mdl_info = ti_summary(mdl, input_size=(1, 1, len(ds.input_cols)),
                                          verbose=0, **info_kwargs)
                elif model_tag == 'tcn':
                    # Conv1D expects (#Batchsize, #Channels, #TimeSeriesLength)
                    mdl_info = ti_summary(mdl, input_size=(1, len(ds.input_cols), tbptt_size),
                                          device=device, verbose=0)

                pbar.write(str(mdl_info))
                logs['model_size'] = mdl_info.total_params
            # it is important to transfer model to CPU right here, after model_stats were printed
            #  otherwise, one model in a process might get back to GPU, whysoever
            mdl.to(device)

            if model_tag == 'node':
                # generate time arrays
                t_span_train = torch.arange(0.0, len(train_tensor), dtype=torch.float32).to(device) / 2
                t_span_test = torch.arange(0.0, len(test_tensor), dtype=torch.float32).to(device) / 2
                if val_tensor is not None:
                    t_span_val = torch.arange(0.0, len(val_tensor), dtype=torch.float32).to(device) / 2
            else:
                # calculate amount of tbptt-len subsequences within a chunk
                n_seqs = np.ceil(train_tensor.shape[0] / tbptt_size).astype(int)

            # generate shuffled indices beforehand
            idx_mat = []
            for i in range(epochs):
                idx = np.arange(train_tensor.shape[1])
                np.random.shuffle(idx)
                idx_mat.append(idx)
            idx_mat = np.vstack(idx_mat)
            batch_size = np.ceil(train_tensor.shape[1] / n_batches).astype(int)

            # Training loop
            for epoch in pbar:
                mdl.train()
                # shuffle profiles
                indices = idx_mat[epoch]
                train_tensor_shuffled = train_tensor[:, indices, :]
                train_sample_weights_shuffled = train_sample_weights[:, indices]
                for n in range(n_batches):
                    # mini-batch training (do not consider all profiles at once)
                    train_tensor_shuffled_n_batched = \
                        train_tensor_shuffled[:, n*batch_size:min((n+1)*batch_size, train_tensor_shuffled.shape[1]), :]
                    train_sample_weights_shuffled_n_batched = \
                        train_sample_weights_shuffled[:,
                                                      n * batch_size:min((n+1)*batch_size,
                                                                         train_sample_weights_shuffled.shape[1])]
                    if model_tag != 'node':
                        if model_tag != 'tcn':
                            hidden = get_initial_hidden_tensor(train_tensor_shuffled_n_batched, model_tag=model_tag,
                                                               n_targets=n_targets, layer_config=layer_cfg_d)
                        for i in range(n_seqs):
                            # iter from beginning to end of subsequence/chunk
                            mdl.zero_grad()
                            if model_tag != 'tcn':
                                if model_tag == 'gru':
                                    hidden = [h.detach() for h in hidden]
                                elif model_tag == 'lstm':
                                    hidden = [[h.detach() for h in hh] for hh in hidden]
                                else:
                                    hidden = hidden.detach()

                            train_sample = train_tensor_shuffled_n_batched[i * tbptt_size:(i+1)*tbptt_size, :, :]
                            sample_w = train_sample_weights_shuffled_n_batched[i*tbptt_size:(i+1)*tbptt_size, :]
                            gtruth_targets = train_sample[:, :, -n_targets:]

                            if continue_mdl is not None:
                                # LBFGS opt
                                def closure():
                                    if torch.is_grad_enabled():
                                        opt.zero_grad()
                                    output, _ = mdl(train_sample[:, :, :len(ds.input_cols)], hidden)
                                    train_loss = loss(output, gtruth_targets, sample_w)
                                    if train_loss.requires_grad:
                                        train_loss.backward()
                                    return train_loss
                                
                                opt.step(closure)  # lbfgs optimization
                                with torch.no_grad(): # for logging
                                    output, hidden = mdl(train_sample[:, :, :len(ds.input_cols)], hidden) # update "hidden"
                                    train_loss = loss(output, gtruth_targets, sample_w)
                            else:
                                # propagate model
                                if model_tag == 'tcn':
                                    output = mdl(train_sample[:, :, :len(ds.input_cols)].permute(1, 2, 0)).permute(2, 0, 1)
                                else:
                                    output, hidden = mdl(train_sample[:, :, :len(ds.input_cols)], hidden)
                                train_loss = loss(output, gtruth_targets, sample_w)
                                train_loss.backward()
                                opt.step()
                    elif model_tag == 'node':
                        opt.zero_grad()
                        mdl.u = train_tensor_shuffled_n_batched[:, :, :len(ds.input_cols)]
                        trajectory = odeint_adjoint(mdl, y0=train_tensor_shuffled_n_batched[0, :, -len(ds.target_cols):],
                                                    t=t_span_train, method='euler', adjoint_method="euler",)

                        train_loss = loss(trajectory, train_tensor_shuffled_n_batched[:, :, -len(ds.target_cols):],
                                          train_sample_weights_shuffled_n_batched)
                        train_loss.backward()
                        opt.step()
                with torch.no_grad():
                    logs["loss_trends_train"][fold_i].append(train_loss.item())
                    pbar_str = f'Loss {train_loss.item():.2e}'
                # validation set
                if val_tensor is not None:
                    with torch.no_grad():
                        mdl.eval()
                        if model_tag != 'node':
                            if model_tag == 'tcn':
                                val_pred = mdl(val_tensor[:, :, :len(ds.input_cols)].permute(1, 2, 0)).permute(2, 0, 1)
                            else:
                                hidden = get_initial_hidden_tensor(val_tensor, model_tag=model_tag,
                                                                   n_targets=n_targets,
                                                                   layer_config=layer_cfg_d)

                                val_pred, hidden = mdl(val_tensor[:, :, :len(ds.input_cols)], hidden)
                            val_loss = loss(val_pred, val_tensor[:, :, -n_targets:],
                                            val_sample_weights).item()
                        elif model_tag == 'node':
                            mdl.u = val_tensor[:, :, :len(ds.input_cols)]
                            val_traj = odeint(mdl, y0=val_tensor[0, :, -len(ds.target_cols):],
                                              t=t_span_val, method="euler")

                            # logging
                            val_loss = loss(val_traj, val_tensor[:, :, -len(ds.target_cols):],
                                            val_sample_weights).item()
                        logs["loss_trends_val"][fold_i].append(val_loss)
                        pbar_str += f'| val loss {val_loss:.2e}'
                pbar.set_postfix_str(pbar_str)
                if np.isnan(val_loss):
                    break
                if half_lr_at is not None:
                    if epoch in half_lr_at:
                        for group in opt.param_groups:
                            group["lr"] *= 0.5

            # test set evaluation
            with torch.no_grad():
                mdl.eval()
                if model_tag != 'node':
                    if model_tag == 'tcn':
                        fold_pred = mdl(test_tensor[:, :, :len(ds.input_cols)].permute(1, 2, 0)).permute(2, 0, 1)
                    else:
                        hidden = get_initial_hidden_tensor(test_tensor, model_tag=model_tag,
                                                           n_targets=n_targets,
                                                           layer_config=layer_cfg_d)
                        fold_pred, hidden = mdl(test_tensor[:, :, :len(ds.input_cols)], hidden)
                    test_traj = fold_pred.cpu().numpy()
                elif model_tag == 'node':
                    mdl.u = test_tensor[:, :, :len(ds.input_cols)]
                    test_traj = odeint(mdl, y0=test_tensor[0, :, -n_targets:],
                                       t=t_span_test, method="euler").numpy()
            # store test set predictions
            for i, tst_set_id in enumerate(sorted(fold_test_profiles)):
                row_mask = all_predictions_df.loc[:, ds.pid] == tst_set_id
                all_predictions_df.loc[row_mask, ds.target_cols] = test_traj[:pid_sizes[tst_set_id], i, :]
            # Save model to logs
            logs["models_state_dict"].append(mdl.state_dict())
            if 'models_arch' not in logs:
                logs["models_arch"] = json.dumps(layer_cfg_d)

        # filter prediction & ground truth placeholders for actual test set
        unrolled_test_profiles = [b for a in test_l for b in a]
        all_predictions_df = all_predictions_df.query(f"{ds.pid} in @unrolled_test_profiles")
        gtruth = gtruth.query(f"{ds.pid} in @unrolled_test_profiles")

        logs['all_predictions_df'] = all_predictions_df.assign(repetition=rep)
        logs['ground_truth'] = gtruth
        logs['end_time'] = pd.Timestamp.now().round(freq='S')
        logs['seed'] = rep
        return logs

    n_seeds = 1 if debug else n_seeds
    print(f"Parallelize over {n_seeds} seeds with {n_jobs} processes..")
    # start experiments in parallel processes
    with Parallel(n_jobs=n_jobs) as prll:
        # list of dicts
        experiment_logs = prll(delayed(run_dyn_training)(s) for s in range(n_seeds))
    dbm = DBManager(model_tag=model_tag, loss_metric='mse', cv=cv_method,
                    scriptname=Path(sys.argv[0]).name,
                    hostname=platform.uname().node, debugmode=debug,
                    dataset=ds, n_folds=1,
                    model_size=experiment_logs[0]['model_size'],
                    model_arch=experiment_logs[0]['models_arch'],
                    comment=run_tag)
    dbm.save(experiment_logs)

    print("done")
    return experiment_logs

def evaluate_generalization(expid, seed, fold):
    """TODO: Write me"""
    
    # query meta info of experiment
    mdl_file = f"exp_{expid}_seed_{seed}_fold_{fold}.pt"
    mdl_path = DBManager.MODEL_DUMP_PATH / mdl_file
    meta_info = DBManager.query(f"Select hostname, model_tag, input_cols, target_cols, layer_cfg from experiments "
                                f"where id = {expid}").iloc[0, :]
    layer_cfg_d = json.loads(meta_info.layer_cfg)
    # tcn special edit
    layer_cfg_d['double_layered'] = False
    input_cols = meta_info.input_cols.strip("}{").split(",")
    target_cols = meta_info.target_cols.strip("}{").split(",")

    # copy model file from remote if necessary
    if not Path.exists(mdl_path):
        # find out host
        host = meta_info.hostname                                 
        print(f"SecureCopy model of experiment {expid} from {host}")
        if host.startswith("n2cn"):
            # pc2
            remote_path = f"/scratch/hpc-prf-neunet/wilhelmk/temp_project_files/projects/mawk-2/data/output/torch_state_dicts/{mdl_file}"
            host = 'noctua2-ln3'
        else:
            remote_path = mdl_path
        subprocess.run(f"scp {host}:{remote_path} {mdl_path}", shell=True)                                  
    
    # create test tensor
    device = torch.device('cpu')
    ds = KaggleDataSet()
    has_ewma = any('_ewma_' in c for c in input_cols)
    has_ewms = any('_ewms_' in c for c in input_cols)
    if any("P_el" in c for c in input_cols):
        fe_extension_scheme = 'extensive'
    elif any("u_s" in c for c in input_cols):
        fe_extension_scheme = 'basic'
    else:
        fe_extension_scheme = 'plain'
    ds.featurize(scheme=fe_extension_scheme)
    if has_ewma or has_ewms:
        ds.data = add_mov_stats(ds.data, input_feats=ds.input_cols, pid=ds.pid, add_ewma=has_ewma, add_ewms=has_ewms)
        ds.input_cols = [c for c in ds.data if c not in ds.target_cols + [ds.pid]]
        ds.data = ds.data.loc[:, ds.input_cols + [ds.pid] + ds.target_cols]  # rearrange
    ds.normalize()
    pid_sizes = ds.get_pid_sizes().to_dict()  # test and val set sizes
    _, _, test_l = ds.get_profiles_for_cv('1fold_static_diss')  # get generalization set
    test_l = test_l[0]  # we have only one fold
    test_tensor, test_sample_weights = generate_tensor(test_l, ds, device)

    # load model from disk
    if meta_info.model_tag.endswith("tnn"):
        mdl_cls = {'tnn': TNNCell, 'capless_tnn': CaplessTNNCell, 'single_sub_tnn': SingleSubNNTNNCell}[meta_info.model_tag]
        mdl = DifferenceEqLayer(mdl_cls, n_targets=len(target_cols), input_feats=input_cols, temperature_cols=ds.temperature_cols,
                                sample_time=ds.sample_time, non_temperature_cols=ds.non_temperature_cols, layer_cfg=layer_cfg_d)
        mdl = torch.jit.script(mdl)  # new syntax as of pytorch 1.2
    elif meta_info.model_tag in ('lstm', 'gru', 'tcn'):
        layer_cls = {'lstm': ResidualLSTM, 'gru': ResidualGRU, 'tcn': TemporalConvNet}[meta_info.model_tag]
        mdl = layer_cls(len(input_cols), layer_cfg=layer_cfg_d)
    mdl = mdl.to(device)
    mdl.load_state_dict(torch.load(mdl_path))
    mdl.eval()

    # predict test set
    with torch.no_grad():
        if meta_info.model_tag == 'tcn':
            fold_pred = mdl(test_tensor[:, :, :len(ds.input_cols)].permute(1, 2, 0)).permute(2, 0, 1)
        else:
            hidden = get_initial_hidden_tensor(test_tensor, model_tag=meta_info.model_tag,
                                                n_targets=len(target_cols),
                                                layer_config=layer_cfg_d)
            fold_pred, hidden = mdl(test_tensor[:, :, :len(ds.input_cols)], hidden)
        test_traj = fold_pred.cpu().numpy()
    all_predictions_df = ds.data.loc[:, ds.target_cols + [ds.pid]]
    gtruth = all_predictions_df.copy()
    for i, tst_set_id in enumerate(sorted(test_l)):
        row_mask = all_predictions_df.loc[:, ds.pid] == tst_set_id
        all_predictions_df.loc[row_mask, ds.target_cols] = test_traj[:pid_sizes[tst_set_id], i, :]
    all_predictions_df = all_predictions_df.query(f"{ds.pid} in @test_l")
    gtruth = gtruth.query(f"{ds.pid} in @test_l")
    metrics = calc_metrics(all_predictions_df.loc[:, ds.target_cols], 
                           gtruth.loc[:, ds.target_cols],
                           target_scale=ds.temperature_scale)
    pprint(metrics)
    # unnormalize
    all_predictions_df.loc[:, ds.target_cols] *= ds.temperature_scale
    ds.data.loc[:, ds.target_cols] *= ds.temperature_scale
    return all_predictions_df, ds, metrics, mdl
