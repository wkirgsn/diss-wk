from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae
from sklearn.model_selection import KFold, GroupKFold
import numpy as np
import pandas as pd
import torch


def calc_metrics(pred, gtruth, target_scale=None):
    """Given predictions and the corresponding ground truth in normalized 
    quantities, return a set of metrics as dictionary"""

    target_scale = target_scale or 100
    if hasattr(pred, 'values'):
        pred = pred.to_numpy()
    if hasattr(gtruth, 'values'):
        gtruth = gtruth.to_numpy()
    if len(pred.shape) == 1:
        # one dim vector convert to column vector
        pred = pred.reshape(-1, 1)
        gtruth = gtruth.reshape(-1, 1)
    assert pred.shape == gtruth.shape, f'pred and gtruth shape do not match ({pred.shape} != {gtruth.shape})'
    pred *= target_scale  # denormalize
    gtruth *= target_scale  # denormalize

    diffs = pred - gtruth

    return {'mse': mse(gtruth, pred),
            'l_infty': np.max(np.abs(diffs)),
            'r2': r2_score(gtruth, pred),
            'mae': mae(gtruth, pred),
            'l_infty_over': np.max(diffs),
            'l_infty_under': np.min(diffs),

            }


class ThermallyWeightedMSE(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, underest_penalty_factor=1,
                 high_temp_exponent=2):
        super().__init__()
        self.underest_penalty_factor = underest_penalty_factor
        self.high_temp_exponent = high_temp_exponent
        self.max_temp = 175 

    def forward(self, inputs, targets, sample_w):
        y_true, y_pred = targets, inputs

        under_penalty = (y_true - y_pred > 0).int() * \
            self.underest_penalty_factor + 1
        high_temp_penalty = (y_true / self.max_temp) ** self.high_temp_exponent
        squared_diffs = torch.nn.functional.mse_loss(
            y_pred, y_true, reduction='none')

        total_weight = sample_w[:, :, None] * under_penalty * high_temp_penalty
        weighted_mse = squared_diffs * total_weight
        weighted_mse = weighted_mse.sum() / sample_w.sum()

        return weighted_mse


class SampleWeightedMSELoss(torch.nn.MSELoss):

    def forward(self, inputs, targets, sample_w):
        y_true, y_pred = targets, inputs

        squared_diffs = torch.nn.functional.mse_loss(
            y_pred, y_true, reduction='none')
        total_weight = sample_w[:, :, None]
        weighted_mse = squared_diffs * total_weight
        weighted_mse = weighted_mse.sum() / sample_w.sum()
        return weighted_mse


def generate_tensor(profiles_list, _ds, device, pid_lbl=None):
    """Returns tensors with shape: (#time steps, #profiles, #features)"""
    pid_lbl = pid_lbl or _ds.pid
    if len(profiles_list) == 0:
        return None, None
    # there are possibly multiple pid columns due to chunked training set
    pid_lbls = [c for c in _ds.data if c.endswith(_ds.pid)]
    # tensor shape: (#time steps, #profiles, #features)
    tensor = np.full((_ds.get_pid_sizes(pid_lbl)[profiles_list].max(),
                      len(profiles_list),
                      _ds.data.shape[1] - len(pid_lbls)), np.nan)
    for i, (pid, df) in enumerate(_ds.data.loc[_ds.data.loc[:, pid_lbl].isin(profiles_list), :]
                                          .groupby(pid_lbl)):
        tensor[:len(df), i, :] = df.drop(columns=pid_lbls).to_numpy()

    sample_weights = 1 - np.isnan(tensor[:, :, 0])

    tensor = np.nan_to_num(tensor).astype(np.float32)
    tensor = torch.from_numpy(tensor).to(device)
    sample_weights = torch.from_numpy(sample_weights).to(device)
    return tensor, sample_weights
