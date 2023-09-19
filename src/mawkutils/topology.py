import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn import Parameter as TorchParam
from torch import Tensor, sigmoid
from typing import List, Tuple, Optional
import numpy as np
from numba import njit, jit
import copy
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1


class SinusAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GeneralizedCosinusUnit(nn.Module):
    def forward(self, x):
        return torch.cos(x)*x


ACTIVATION_FUNCS = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU,
                    'biased_elu': Biased_Elu, 'sinus': SinusAct, 'gcu': GeneralizedCosinusUnit}


class DifferenceEqLayer(nn.Module):
    """For discretized ODEs"""

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        device = torch.device('cpu')
        self.cell = cell(*cell_args, **cell_kwargs).to(device)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class MultiLayerResidualRecursive(nn.Module):

    def __init__(self, cell_cls, input_size, layer_cfg=None):
        super().__init__()
        device = torch.device('cpu')
        layer_cfg = layer_cfg or {'f': [{'units': 8}, {'units': 4}]}
        self.n_layers = len(layer_cfg['f'])
        self.cells = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.residual = layer_cfg.get('residual', False)
        n_units = input_size
        for layer_specs in layer_cfg['f']:
            self.cells.append(cell_cls(input_size=n_units,
                                       hidden_size=layer_specs['units']).to(device))
            if self.residual:
                if n_units == layer_specs['units']:
                    shortcut = nn.Identity()
                else:
                    shortcut = nn.Linear(in_features=n_units,
                                         out_features=layer_specs['units']).to(device)
                self.shortcuts.append(shortcut)
            n_units = layer_specs['units']


class ResidualLSTM(MultiLayerResidualRecursive):
    def __init__(self, *args, **kwargs):
        super().__init__(nn.LSTMCell, *args, **kwargs)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        hx, cx = state
        # state - (#n_layers, #batch_size, #hidden_size)
        for i in range(len(inputs)):
            inp = inputs[i]
            for j, cell in enumerate(self.cells):
                out, cx[j] = cell(inp, (hx[j], cx[j]))  # zero state at beginning
                hx[j] = out
                if self.residual:
                    out += self.shortcuts[j](inp)
                    out = torch.clip(out, -10, 10)
                inp = out
            outputs += [out]
        return torch.stack(outputs), (hx, cx)


class ResidualGRU(MultiLayerResidualRecursive):
    def __init__(self, *args, **kwargs):
        super().__init__(nn.GRUCell, *args, **kwargs)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        # state - (#n_layers, #batch_size, #hidden_size)
        for i in range(len(inputs)):
            inp = inputs[i]
            for j, cell in enumerate(self.cells):
                out = cell(inp, state[j])  # zero state or gtruth at beginning
                state[j] = out
                if self.residual:
                    out += self.shortcuts[j](inp)
                    out = torch.clip(out, -10, 10)
                inp = out
            outputs += [out]
        return torch.stack(outputs), state


class ExplEulerCell(nn.Module):
    def __init__(self, n_targets, input_feats,
                 temperature_cols, sample_time=0.5, layer_cfg=None):
        super().__init__()

        self.sample_time = sample_time  # in s
        self.output_size = n_targets

        # layer config init
        layer_default = {
            'f': [{'units': 32, 'activation': 'relu'},
                  {'units': n_targets, 'activation': 'tanh'}],
        }
        self.layer_cfg = layer_cfg or layer_default

        # main sub NN
        f_layers = []
        f_units = len(input_feats) + n_targets

        for layer_specs in self.layer_cfg['f']:
            lay = nn.Linear(f_units, layer_specs["units"])
            lay.weight.data.normal_(0, 1e-2)
            lay.bias.data.normal_(0, 1e-2)
            f_layers.append(lay)
            if layer_specs.get('activation', 'linear') != 'linear':
                f_layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            f_units = layer_specs["units"]
        self.f = nn.Sequential(*f_layers)

        self.temp_idcs: List[int] = [i for i, x in enumerate(
            input_feats) if x in temperature_cols]

    def forward(self, inp, hidden):
        prev_out = hidden
        all_input = torch.cat([inp, hidden], dim=1)
        out = prev_out + self.sample_time * self.f(all_input)
        return prev_out, out


class TNNCell(nn.Module):
    def __init__(self, n_targets, input_feats,
                 temperature_cols, non_temperature_cols,
                 n_virtual_temperatures=0, sample_time=0.5, layer_cfg=None):
        super().__init__()
        self.layer_cfg = layer_cfg or jku_default
        self.drop_g = 'drop_g' in self.layer_cfg.keys()
        self.sample_time = sample_time  # in s
        self.output_size = n_targets
        virt_ext_output_size = n_targets + n_virtual_temperatures
        n_temps = len(temperature_cols) + n_virtual_temperatures
        n_conds = int(0.5 * n_temps * (n_temps - 1))
        if self.drop_g:
            n_conds -= len(self.layer_cfg['drop_g'])
        # layer config init
        node_default = {
            'p': [{'units': 4, 'activation': 'tanh'},
                  {'units': virt_ext_output_size, 'activation': 'sigmoid'}],
            'g': [{'units': 2, 'activation': 'tanh'},
                  {'units': n_conds, 'activation': 'biased_elu'}]
        }
        jku_default = {
            'p': [{'units': 8, 'activation': 'tanh'},
                  {'units': virt_ext_output_size, 'activation': 'linear'}],
            'g': [{'units': n_conds, 'activation': 'sigmoid'}]
        }

        if self.drop_g:
            # omit thermal conductance pairs
            #  Attention! the order of arg "temperature_cols" must align with "temps"
            #  in forward() below. This is partly determined by the Dataset class instance
            #  that is used with this TNN model
            drop_g = []
            for pair in self.layer_cfg.get('drop_g', []):
                # create two fancy indexing arrays
                c1, c2 = pair.split(',')
                # make sure to have drop_g in layer_cfg set such that input temperature cols
                #  are not named first, as these rows are cropped in the adjacency matrix later
                c1_idx, c2_idx = temperature_cols.index(c1), temperature_cols.index(c2)
                assert c1_idx < c2_idx, \
                    f"{c1} appears later than {c2} in temperature_cols, dropping would have no effect"
                drop_g.append([c1_idx, c2_idx])
            drop_g = np.array(drop_g)
            self.drop_rows, self.drop_cols = drop_g[:, 0], drop_g[:, 1]

        # populate adjacency matrix

        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        adj_idx_arr = np.ones_like(self.adj_mat)
        if self.drop_g:
            adj_idx_arr[self.drop_rows, self.drop_cols] = 0
            self.drop_rows = torch.from_numpy(self.drop_rows).type(torch.long)
            self.drop_cols = torch.from_numpy(self.drop_cols).type(torch.long)
        else:
            # this is merely to satisfy torch jit script
            self.drop_rows = torch.from_numpy(np.zeros(2)).type(torch.long)
            self.drop_cols = torch.from_numpy(np.zeros(2)).type(torch.long)
        triu_idx = np.triu_indices(n_temps, 1)
        adj_idx_arr = adj_idx_arr[triu_idx].ravel()
        self.adj_mat[triu_idx] = np.cumsum(adj_idx_arr) - 1
        self.adj_mat += self.adj_mat.T
        self.adj_mat = torch.from_numpy(self.adj_mat[:self.output_size, :])  # crop
        self.n_temps = n_temps

        # conductances
        g_layers = []
        g_units = len(input_feats) + virt_ext_output_size
        for layer_specs in self.layer_cfg['g']:
            g_layers.append(nn.Linear(g_units, layer_specs["units"]))
            if layer_specs['activation'] != 'linear':
                g_layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            g_units = layer_specs["units"]
        self.conductance_net = nn.Sequential(*g_layers)

        # power losses
        p_layers = []
        p_units = len(input_feats) + virt_ext_output_size
        for layer_specs in self.layer_cfg['p']:
            p_layers.append(nn.Linear(p_units, layer_specs["units"]))
            if layer_specs['activation'] != 'linear':
                p_layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            p_units = layer_specs["units"]
        self.ploss = nn.Sequential(*p_layers)

        # inverse thermal capacitances
        self.caps = TorchParam(torch.Tensor(virt_ext_output_size))
        # nn.init.normal_(self.caps, mean=-7, std=0.5)  # node paper
        if "cap" not in self.layer_cfg:
            nn.init.normal_(self.caps, mean=-9.2, std=0.5)
        else:
            for t, init_mean in zip(self.caps, self.layer_cfg["cap"]):
                nn.init.normal_(t, mean=init_mean, std=0.5)

        # optimized indexing (faster computation)
        self.temp_idcs: List[int] = [i for i, x in enumerate(
            input_feats) if x in temperature_cols]
        self.nontemp_idcs: List[int] = [i for i, x in enumerate(
            input_feats) if x in non_temperature_cols]

        self.debug_info = []

    def forward(self, inp, hidden):
        prev_out = hidden[:, :self.output_size]
        temps = torch.cat([hidden, inp[:, self.temp_idcs]], dim=1)
        all_input = torch.cat([inp, hidden], dim=1)
        conducts = torch.abs(self.conductance_net(all_input))
        power_loss = torch.abs(self.ploss(all_input))
        G_mat = conducts[:, self.adj_mat]
        if self.drop_g:
            G_mat[:, self.drop_rows, self.drop_cols] = 0
        temp_diffs_weighted = torch.sum(
            (temps.unsqueeze(1) - hidden.unsqueeze(-1)) * G_mat, dim=-1)
        inverse_caps = torch.exp(self.caps) 
        out = hidden + self.sample_time * inverse_caps * (temp_diffs_weighted + power_loss)
        return prev_out, torch.clip(out, -1, 3)


class CaplessTNNCell(TNNCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.caps
        for subnet in (self.conductance_net, self.ploss):
            for lay in subnet:
                if isinstance(lay, nn.Linear):
                    lay.weight.data.normal_(0, 1e-2)

    def forward(self, inp, hidden):
        prev_out = hidden[:, :self.output_size]
        temps = torch.cat([hidden, inp[:, self.temp_idcs]], dim=1)
        all_input = torch.cat([inp, hidden], dim=1)
        conducts = torch.abs(self.conductance_net(all_input))
        power_loss = torch.abs(self.ploss(all_input))
        temp_diffs_weighted = torch.sum(
            (temps.unsqueeze(1) - hidden.unsqueeze(-1)) * conducts[:, self.adj_mat], dim=-1)

        out = hidden + self.sample_time * (temp_diffs_weighted + power_loss)
        return prev_out, torch.clip(out, -1, 3)


class SingleSubNNTNNCell(TNNCell):
    def __init__(self, n_targets, input_feats,
                 temperature_cols, non_temperature_cols,
                 n_virtual_temperatures=0, sample_time=0.5, layer_cfg=None):
        super().__init__(n_targets, input_feats, temperature_cols, non_temperature_cols,
                         n_virtual_temperatures, sample_time, layer_cfg)
        del self.conductance_net
        del self.ploss
        f_layers = []
        f_units = len(input_feats) + self.output_size
        n_out_neurons = self.layer_cfg['g'][-1]['units'] + self.layer_cfg['p'][-1]['units']
        for lay_i, layer_specs in enumerate(self.layer_cfg['g']):
            if lay_i + 1 == len(self.layer_cfg['g']):
                # last layer
                layer_specs["units"] = n_out_neurons  # overwrite
            f_layers.append(nn.Linear(f_units, layer_specs["units"]))
            if layer_specs['activation'] != 'linear':
                f_layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            f_units = layer_specs["units"]
        self.f_net = nn.Sequential(*f_layers)

    def forward(self, inp, hidden):
        prev_out = hidden[:, :self.output_size]
        temps = torch.cat([hidden, inp[:, self.temp_idcs]], dim=1)
        all_input = torch.cat([inp, hidden], dim=1)
        f_out = torch.abs(self.f_net(all_input))
        conducts = f_out[:, :-self.output_size]
        power_loss = f_out[:, -self.output_size:]
        G_mat = conducts[:, self.adj_mat]
        if self.drop_g:
            G_mat[:, self.drop_rows, self.drop_cols] = 0
        temp_diffs_weighted = torch.sum(
            (temps.unsqueeze(1) - hidden.unsqueeze(-1)) * G_mat, dim=-1)
        inverse_caps = torch.exp(self.caps)

        out = hidden + self.sample_time * inverse_caps * (temp_diffs_weighted + power_loss)
        return prev_out, torch.clip(out, -1, 3)


class WallscheidLPTN:
    """Numpy + numba functions, no AD"""
    caps0 = 1.0666e4
    caps1 = 6.5093e3
    caps2 = 0.437127e3
    caps3 = 3.5105e3
    const_Rs_sy_sw = 0.0375
    const_Rs_sy_st = 0.0707
    const_Rs_sw_st = 0.0899
    lin_Rs_slope = -54e-4
    lin_Rs_bias = 18e-3
    exp_Rs_magn0 = 1.7275
    exp_Rs_magn1 = 0.8486
    exp_Rs_magn2 = 0.6349
    exp_Rs_b0 = 0.1573
    exp_Rs_b1 = 0.1428
    exp_Rs_b2 = 0.1184
    exp_Rs_a0 = 0.3039
    exp_Rs_a1 = 0.2319
    exp_Rs_a2 = 0.1205
    bipoly_Rs_magn = 0.3528
    bipoly_Rs_a = -0.2484
    bipoly_Rs_b = 0.0276
    bipoly_Rs_c = 0.3331
    ploss_Rdc = 14.6e-3
    ploss_alpha_cu = 20e-4
    ploss_alpha_ac_1 = 0.562
    ploss_alpha_ac_2 = 0.2407
    ploss_beta_cu = 2.5667
    ploss_k_1_0 = 0.5441
    ploss_k_1_1 = 78e-4
    ploss_k_1_2 = 0.0352
    ploss_k_1_3 = -0.7438
    ploss_k_2 = 0.8655
    ploss_alpha_fe = -28e-4
    schlepp_factor = 1.4762

    all_data_mse = 2.9651712310
    train_data_mse = 3.0010848910756165
    input_cols = ["ambient", "coolant", "motor_speed", "i_s",
                  "iron_loss", "schlepp"]

    def __init__(self, target_cols, coeffs_list=None):
        self.target_cols = target_cols
        self.n_targets = self.output_size = len(target_cols)
        if coeffs_list is not None:
            self.coeffs_list = coeffs_list
        else:
            self.coeffs_list = \
                (self.caps0, self.caps1, self.caps2, self.caps3, self.const_Rs_sy_sw, self.const_Rs_sy_st, self.const_Rs_sw_st, self.lin_Rs_slope, self.lin_Rs_bias,
                    self.exp_Rs_magn0, self.exp_Rs_magn1, self.exp_Rs_magn2,
                    self.exp_Rs_b0, self.exp_Rs_b1, self.exp_Rs_b2, self.exp_Rs_a0, self.exp_Rs_a1, self.exp_Rs_a2,
                    self.bipoly_Rs_magn, self.bipoly_Rs_a, self.bipoly_Rs_b, self.bipoly_Rs_c,
                    self.ploss_Rdc, self.ploss_alpha_cu, self.ploss_alpha_ac_1, self.ploss_alpha_ac_2, self.ploss_beta_cu, self.ploss_k_1_0,
                    self.ploss_k_1_1, self.ploss_k_1_2, self.ploss_k_1_3, self.ploss_k_2, self.ploss_alpha_fe, self.schlepp_factor)
        self.track = {}

    def solve_ivp(self, inp_df, Ts=0.5):
        """Explicit Euler with Numba"""
        # construct matrix
        n_targets = self.n_targets
        inp_arr = np.zeros((len(inp_df), n_targets+6))
        inp_arr[0, :n_targets] = inp_df.loc[0, self.target_cols].to_numpy()
        inp_arr[:, n_targets:] = inp_df.loc[:, self.input_cols].to_numpy()

        sol = self.numba_solve_ivp(inp_arr, n_targets, self.coeffs_list, Ts=Ts)
        return sol

    def solve_ivp_pl(self, inp_df, Ts=0.5):
        """Polar version of solve_ivp"""
        n_targets = self.n_targets
        inp_arr = np.zeros((len(inp_df), n_targets+6))
        inp_arr[0, :n_targets] = inp_df.select(self.target_cols).limit(1).to_numpy()
        inp_arr[:, n_targets:] = inp_df.select(["ambient", "coolant", "motor_speed", "i_s",
                                                "iron_loss", "schlepp"]).to_numpy()

        sol = self.numba_solve_ivp(inp_arr, n_targets, self.coeffs_list, Ts=Ts)
        return sol

    @staticmethod
    @njit
    def numba_solve_ivp(inp_arr, n_targets, coeffs, Ts=0.5):
        """Solve initial value problem (explicit Euler)"""
        for i in range(inp_arr.shape[0]-1):
            x = inp_arr[i, :n_targets]
            x_k1 = inp_arr[i+1, :n_targets]
            fast_dxdt(inp_arr[i, :], coeffs, x_k1)
            x_k1[:] = x_k1 * Ts + x
        return inp_arr[:, :n_targets]


@njit
def fast_dxdt(inp_vec, coeffs, out):
    # Source:
    #  Implementation: https://github.com/wkirgsn/thermal-nn/blob/main/aux/lptn_model.py
    #  Theory: https://ris.uni-paderborn.de/record/29616

    # order is important
    theta_pm, theta_sy, theta_st, theta_sw, theta_ambient, theta_coolant, speed, current, measured_loss, schlepp = \
        inp_vec  # denormalized
    caps0, caps1, caps2, caps3, const_Rs_sy_sw, const_Rs_sy_st, const_Rs_sw_st, lin_Rs_slope, lin_Rs_bias, \
        exp_Rs_magn0, exp_Rs_magn1, exp_Rs_magn2, \
        exp_Rs_b0, exp_Rs_b1, exp_Rs_b2, exp_Rs_a0, exp_Rs_a1, exp_Rs_a2,\
        bipoly_Rs_magn, bipoly_Rs_a, bipoly_Rs_b, bipoly_Rs_c, \
        ploss_Rdc, ploss_alpha_cu, ploss_alpha_ac_1, ploss_alpha_ac_2, ploss_beta_cu, ploss_k_1_0, \
        ploss_k_1_1, ploss_k_1_2, ploss_k_1_3, ploss_k_2, ploss_alpha_fe, schlepp_factor = coeffs
    caps = np.array((caps0, caps1, caps2, caps3))
    r_c_sy = lin_Rs_bias * (1 + lin_Rs_slope * (theta_coolant - 20))
    speed_norm = np.abs(speed) / 6000
    r_st_pm = expon_resistance(speed_norm, exp_Rs_magn0, exp_Rs_b0, exp_Rs_a0)
    r_sw_pm = expon_resistance(speed_norm, exp_Rs_magn1, exp_Rs_b1, exp_Rs_a1)
    r_pm_amb = expon_resistance(speed_norm, exp_Rs_magn2, exp_Rs_b2, exp_Rs_a2)

    theta_coolant_norm = theta_coolant / 100
    r_pm_c = np.maximum(bipoly_Rs_magn + bipoly_Rs_a * speed_norm +
                        bipoly_Rs_b * theta_coolant_norm +
                        bipoly_Rs_c * speed_norm * theta_coolant_norm,
                        1e-6)
    current = current / np.sqrt(2)  # need to be scaled for Wallscheid LPTN
    ploss_Rac = ploss_Rdc * (1 + ploss_alpha_ac_1 * speed_norm + ploss_alpha_ac_2 * (speed_norm**2))
    lin_ploss_sw = (1 + ploss_alpha_cu * (theta_sw - 70))
    ploss_dc_ref = 3 * ploss_Rdc * (current**2)
    r_ac_over_dc_m1 = ploss_Rac / ploss_Rdc - 1
    ploss_cu_sw_ref = ploss_dc_ref * (1 + r_ac_over_dc_m1)

    ploss_sw = ploss_dc_ref * lin_ploss_sw + ploss_cu_sw_ref * (r_ac_over_dc_m1 / np.maximum(np.maximum(np.abs(lin_ploss_sw), 1e-4) ** ploss_beta_cu, 1e-5))

    # LUT consists of iron loss, copper loss and mechanical loss
    ploss_fe = measured_loss - schlepp_factor * schlepp - ploss_cu_sw_ref
    normed_current = current / 256
    k1 = np.minimum(np.maximum(ploss_k_1_0 +
                               ploss_k_1_1 * speed_norm +
                               ploss_k_1_2 * normed_current +
                               ploss_k_1_3 * speed_norm * normed_current,
                               0), 1)
    ploss_pm = (1 - k1) * ploss_fe * (1 + ploss_alpha_fe * (theta_pm - 63))
    ploss_sy = ploss_k_2 * k1 * ploss_fe * (1 + ploss_alpha_fe * (theta_sy - 55))
    ploss_st = (1 - ploss_k_2) * k1 * ploss_fe * (1 + ploss_alpha_fe * (theta_st - 65))

    ploss_temps = np.array((ploss_pm, ploss_sy, ploss_st, ploss_sw))
    pm_diffs = (theta_st - theta_pm) / r_st_pm + \
        (theta_sw - theta_pm) / r_sw_pm + \
        (theta_ambient - theta_pm) / r_pm_amb + \
        (theta_coolant - theta_pm) / r_pm_c
    sy_diffs = (theta_sw - theta_sy) / const_Rs_sy_sw + \
        (theta_st - theta_sy) / const_Rs_sy_st + \
        (theta_coolant - theta_sy) / r_c_sy
    st_diffs = (theta_sy - theta_st) / const_Rs_sy_st + \
        (theta_sw - theta_st) / const_Rs_sw_st + \
        (theta_pm - theta_st) / r_st_pm
    sw_diffs = (theta_sy - theta_sw) / const_Rs_sy_sw + \
        (theta_st - theta_sw) / const_Rs_sw_st + \
        (theta_pm - theta_sw) / r_sw_pm
    temp_diffs = np.array((pm_diffs, sy_diffs, st_diffs, sw_diffs))
    out[:] = (temp_diffs + ploss_temps) / caps


@njit
def expon_resistance(speed, magn, expo, summ):
    return magn * np.exp(-speed / expo) + summ


class LPTNTorch(nn.Module):
    """Source:
    Implementation: https://github.com/wkirgsn/thermal-nn/blob/main/aux/lptn_model.py
    Theory: https://ris.uni-paderborn.de/record/29616
    """

    coeffs_d = dict(
        caps0 = 1.0666e4,
        caps1 = 6.5093e3,
        caps2 = 0.437127e3,
        caps3 = 3.5105e3,
        const_Rs_sy_sw = 0.0375,
        const_Rs_sy_st = 0.0707,
        const_Rs_sw_st = 0.0899,
        lin_Rs_slope = -54e-4,
        lin_Rs_bias = 18e-3,
        exp_Rs_magn0 = 1.7275,
        exp_Rs_magn1 = 0.8486,
        exp_Rs_magn2 = 0.6349,
        exp_Rs_b0 = 0.1573,
        exp_Rs_b1 = 0.1428,
        exp_Rs_b2 = 0.1184,
        exp_Rs_a0 = 0.3039,
        exp_Rs_a1 = 0.2319,
        exp_Rs_a2 = 0.1205,
        bipoly_Rs_magn = 0.3528,
        bipoly_Rs_a = -0.2484,
        bipoly_Rs_b = 0.0276,
        bipoly_Rs_c = 0.3331,
        ploss_Rdc = 14.6e-3,
        ploss_alpha_cu = 20e-4,
        ploss_alpha_ac_1 = 0.562,
        ploss_alpha_ac_2 = 0.2407,
        ploss_beta_cu = 2.5667,
        ploss_k_1_0 = 0.5441,
        ploss_k_1_1 = 78e-4,
        ploss_k_1_2 = 0.0352,
        ploss_k_1_3 = -0.7438,
        ploss_k_2 = 0.8655,
        ploss_alpha_fe = -28e-4,
        schlepp_factor = 1.4762
    )

    def __init__(self, target_cols, input_cols, temp_scale=200):
        nn.Module.__init__(self)
        self.target_cols = target_cols
        self.n_targets = self.output_size = len(target_cols)
        self.coeffs = TorchParam(torch.Tensor(len(self.coeffs_d)), requires_grad=True)
        nn.init.normal_(self.coeffs[:4], mean=-7, std=1)
        self.input_indices = [input_cols.index(c) for c in WallscheidLPTN.input_cols]
        self.sample_time = 0.5
        self.temp_scale = temp_scale

    def expon_resistance(self, speed, magn, expo, summ):
        return magn * torch.exp(-speed / expo) + summ

    def softplus(self, a):
        return torch.log(1 + torch.exp(a))

    def forward(self, inp, hidden):
        "WallscheidLPTN Logic but with learnable constants, constrained by appropriate activation functions"

        theta_pm, theta_sy, theta_st, theta_sw = torch.tensor_split(hidden, hidden.shape[1], dim=1)
        theta_ambient, theta_coolant, speed, current, iron_loss, schlepp = \
            torch.tensor_split(inp[:, self.input_indices], len(self.input_indices), dim=1)
        caps0, caps1, caps2, caps3, const_Rs_sy_sw, const_Rs_sy_st, const_Rs_sw_st, lin_Rs_slope, lin_Rs_bias, \
            exp_Rs_magn0, exp_Rs_magn1, exp_Rs_magn2, \
            exp_Rs_b0, exp_Rs_b1, exp_Rs_b2, exp_Rs_a0, exp_Rs_a1, exp_Rs_a2,\
            bipoly_Rs_magn, bipoly_Rs_a, bipoly_Rs_b, bipoly_Rs_c, \
            ploss_Rdc, ploss_alpha_cu, ploss_alpha_ac_1, ploss_alpha_ac_2, ploss_beta_cu, ploss_k_1_0, \
            ploss_k_1_1, ploss_k_1_2, ploss_k_1_3, ploss_k_2, ploss_alpha_fe, schlepp_factor = \
            torch.tensor_split(self.coeffs, self.coeffs.shape[0])
        inverse_caps = torch.cat([caps0, caps1, caps2, caps3])

        # torch/AD specific transforms
        #  belus/softplus
        inverse_caps = self.softplus(inverse_caps)
        const_Rs_sy_sw = self.softplus(const_Rs_sy_sw)
        const_Rs_sy_st = self.softplus(const_Rs_sy_st)
        const_Rs_sw_st = self.softplus(const_Rs_sw_st)
        lin_Rs_bias = self.softplus(lin_Rs_bias)
        bipoly_Rs_magn = self.softplus(bipoly_Rs_magn)
        ploss_Rdc = self.softplus(ploss_Rdc)
        lin_Rs_slope = -self.softplus(lin_Rs_slope)
        ploss_alpha_cu = self.softplus(ploss_alpha_cu)
        exp_Rs_magn0 = self.softplus(exp_Rs_magn0)
        exp_Rs_magn1 = self.softplus(exp_Rs_magn1)
        exp_Rs_magn2 = self.softplus(exp_Rs_magn2)
        exp_Rs_b0 = self.softplus(exp_Rs_b0)
        exp_Rs_b1 = self.softplus(exp_Rs_b1)
        exp_Rs_b2 = self.softplus(exp_Rs_b2)
        exp_Rs_a0 = self.softplus(exp_Rs_a0)
        exp_Rs_a1 = self.softplus(exp_Rs_a1)
        exp_Rs_a2 = self.softplus(exp_Rs_a2)
        ploss_beta_cu = self.softplus(ploss_beta_cu)
        schlepp_factor = self.softplus(schlepp_factor)

        # 0,1 or -1,1
        bipoly_Rs_a = -torch.sigmoid(bipoly_Rs_a)
        bipoly_Rs_b = -torch.sigmoid(bipoly_Rs_b)
        bipoly_Rs_c = torch.tanh(bipoly_Rs_c)
        ploss_k_2 = torch.sigmoid(ploss_k_2)
        ploss_alpha_fe = -torch.sigmoid(ploss_alpha_fe)*1e-2

        # copy from fast_dxdt (see above)
        r_c_sy = lin_Rs_bias * (1 + lin_Rs_slope * (theta_coolant - 20/self.temp_scale))
        speed_norm = torch.abs(speed)  # / 6000 already normalized
        # self.expon_resistance(speed_norm, exp_Rs_magn0, exp_Rs_b0, exp_Rs_a0)
        g_st_pm = exp_Rs_magn0 * torch.sigmoid(speed * exp_Rs_b0) + exp_Rs_a0
        # self.expon_resistance(speed_norm, exp_Rs_magn1, exp_Rs_b1, exp_Rs_a1)
        g_sw_pm = exp_Rs_magn1 * torch.sigmoid(speed * exp_Rs_b1) + exp_Rs_a1
        # self.expon_resistance(speed_norm, exp_Rs_magn2, exp_Rs_b2, exp_Rs_a2)
        g_pm_amb = exp_Rs_magn2 * torch.sigmoid(speed * exp_Rs_b2) + exp_Rs_a2

        theta_coolant_norm = theta_coolant * 2  # / 100
        g_pm_c = torch.clip(bipoly_Rs_magn + bipoly_Rs_a * speed_norm +
                            bipoly_Rs_b * theta_coolant_norm +
                            bipoly_Rs_c * speed_norm * theta_coolant_norm,
                            1e-6, None)
        # current = current #/ np.sqrt(2)  # need to be scaled for Wallscheid LPTN
        lin_ploss_sw = (1 + ploss_alpha_cu * (theta_sw - 70/self.temp_scale))
        ploss_dc_ref = 3 * ploss_Rdc * (current**2)
        r_ac_over_dc_m1 = ploss_alpha_ac_1 * speed_norm + ploss_alpha_ac_2 * (speed_norm**2)
        ploss_cu_sw_ref = ploss_dc_ref * (1 + r_ac_over_dc_m1)

        ploss_sw = ploss_dc_ref * lin_ploss_sw + ploss_cu_sw_ref * \
            (r_ac_over_dc_m1 / torch.clip(torch.clip(torch.abs(lin_ploss_sw), min=1e-4) ** ploss_beta_cu, min=1e-5))

        # LUT consists of iron loss, copper loss and mechanical loss
        ploss_fe = iron_loss - schlepp_factor * schlepp - ploss_cu_sw_ref  # is this still valid with normalized values?
        normed_current = current  # / 256
        k1 = torch.sigmoid(ploss_k_1_0 + ploss_k_1_1 * speed_norm +
                            ploss_k_1_2 * normed_current +
                            ploss_k_1_3 * speed_norm * normed_current)
        ploss_pm = (1 - k1) * ploss_fe * (1 + ploss_alpha_fe * (theta_pm - 63/self.temp_scale))
        ploss_sy = ploss_k_2 * k1 * ploss_fe * (1 + ploss_alpha_fe * (theta_sy - 55/self.temp_scale))
        ploss_st = (1 - ploss_k_2) * k1 * ploss_fe * (1 + ploss_alpha_fe * (theta_st - 65/self.temp_scale))

        ploss_temps = torch.cat((ploss_pm, ploss_sy, ploss_st, ploss_sw), dim=1)
        pm_diffs = (theta_st - theta_pm) * g_st_pm + \
            (theta_sw - theta_pm) * g_sw_pm + \
            (theta_ambient - theta_pm) * g_pm_amb + \
            (theta_coolant - theta_pm) * g_pm_c
        sy_diffs = (theta_sw - theta_sy) * const_Rs_sy_sw + \
            (theta_st - theta_sy) * const_Rs_sy_st + \
            (theta_coolant - theta_sy) * r_c_sy
        st_diffs = (theta_sy - theta_st) * const_Rs_sy_st + \
            (theta_sw - theta_st) * const_Rs_sw_st + \
            (theta_pm - theta_st) * g_st_pm
        sw_diffs = (theta_sy - theta_sw) * const_Rs_sy_sw + \
            (theta_st - theta_sw) * const_Rs_sw_st + \
            (theta_pm - theta_sw) * g_sw_pm
        temp_diffs = torch.cat((pm_diffs, sy_diffs, st_diffs, sw_diffs), dim=1)

        out = hidden + self.sample_time * inverse_caps * (temp_diffs + ploss_temps)
        return hidden, torch.clip(out, -1, 3)


class AdjointConformTNN(nn.Module):

    def __init__(self, u, n_targets, input_feats,
                 temperature_cols, non_temperature_cols,
                 n_virtual_temperatures=0):
        super().__init__()

        self.output_size = n_targets
        virt_ext_output_size = n_targets + n_virtual_temperatures

        # inverse thermal capacitances
        self.caps = TorchParam(torch.Tensor(virt_ext_output_size))
        nn.init.normal_(self.caps, mean=-7, std=0.5)

        # therm. conductances
        n_temps = len(temperature_cols) + n_virtual_temperatures
        n_conds = int(0.5 * n_temps * (n_temps - 1))
        self.conductance_net = nn.Sequential(
            nn.Linear(len(input_feats) + virt_ext_output_size, 2),
            nn.Tanh(),
            nn.Linear(2, n_conds),
            Biased_Elu()
        )

        # populate adjacency matrix
        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        adj_idx_arr = np.ones_like(self.adj_mat)
        triu_idx = np.triu_indices(n_temps, 1)
        adj_idx_arr = adj_idx_arr[triu_idx].ravel()
        self.adj_mat[triu_idx] = np.cumsum(adj_idx_arr) - 1
        self.adj_mat += self.adj_mat.T
        self.adj_mat = torch.from_numpy(self.adj_mat)
        self.n_temps = n_temps

        # power losses
        self.ploss = nn.Sequential(
            nn.Linear(len(input_feats) + virt_ext_output_size, 4),
            nn.Tanh(),
            nn.Linear(4, virt_ext_output_size),
            nn.Sigmoid()
        )

        self.temp_idcs = [i for i, x in enumerate(
            input_feats) if x in temperature_cols]
        self.nontemp_idcs = [i for i, x in enumerate(
            input_feats) if x in non_temperature_cols]

        # optimized indexing (faster computation)
        self.temps_indexer = np.array([j for i in range(virt_ext_output_size)
                                       for j in range(n_temps) if j != i]).reshape(-1, n_temps-1)
        self.adj_mat_indexed = self.adj_mat[np.repeat(np.arange(virt_ext_output_size), n_temps-1),
                                            self.temps_indexer.ravel()].reshape(virt_ext_output_size, -1).tolist()
        self.temps_indexer = self.temps_indexer.tolist()

        self.u = u

    def forward(self, t, x):

        # plant input
        inp = torch.FloatTensor(
            self.u[(t.detach().numpy()*2).astype(int), :, :])
        # print(inp.shape)
        temps = torch.cat([x, inp[:, self.temp_idcs]], dim=1)
        all_input = torch.cat([inp, x], dim=1)
        conducts = self.conductance_net(all_input)
        power_loss = torch.abs(self.ploss(all_input))

        temp_diffs = torch.sum(
            (temps[:, self.temps_indexer] - x.unsqueeze(-1)) *
            conducts[:, self.adj_mat_indexed],
            dim=-1)
        out = torch.exp(self.caps) * (temp_diffs + power_loss)

        return torch.clip(out, -1, 1)


class MLP(nn.Module):
    def __init__(self, n_inputs, layer_cfg=None):
        super().__init__()
        self.layer_cfg = layer_cfg or {'f': [{'units': 16, 'activation': 'relu'},
                                             {'units': 4, 'activation': 'linear'}]}
        # build according to layer_cfg
        layers = []
        units = n_inputs
        for layer_specs in self.layer_cfg['f']:
            layers.append(nn.Linear(units, layer_specs["units"]))
            if layer_specs['activation'] != 'linear':
                layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            units = layer_specs["units"]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# TCN stuff
# from: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 residual=True, double_layered=True, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                                    padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout)
        if double_layered:
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation))
            self.dropout2 = nn.Dropout1d(dropout)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()
        self.residual = residual
        if residual:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        else:
            self.downsample = None
        self.double_layered = double_layered
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.double_layered:
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            y = torch.clip(out + res, -10, 10)  # out += res is not allowed, it would become an inplace op, weird
        else:
            y = out
        return self.relu(y)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, layer_cfg=None):
        super(TemporalConvNet, self).__init__()
        layer_cfg = layer_cfg or {'f': [{'units': 8}, {'units': 4}]}
        layers = []
        dilation_offset = layer_cfg.get("starting_dilation_rate", 0)  # >= 0
        for i, l_cfg in enumerate(layer_cfg['f']):
            kernel_size = l_cfg.get('kernel_size', 3)
            dilation_size = 2 ** (i + dilation_offset)
            in_channels = num_inputs if i == 0 else layer_cfg['f'][i-1]['units']
            layers += [TemporalBlock(in_channels, l_cfg['units'], kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, residual=layer_cfg.get('residual', True),
                                     double_layered=layer_cfg.get("double_layered", True),
                                     dropout=layer_cfg.get("dropout", 0.2)),
                       ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
