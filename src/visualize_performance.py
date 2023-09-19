"""Given an experiment id (and seed), visualize the performance"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import subprocess
from mawkutils.data import DBManager as DBM, KaggleDataSet
from mawkutils.render import prettify


PROC_DATA_PATH = Path.cwd().parent / 'data' / 'processed'
STATEDICT_PATH = Path().cwd().parent / 'models' / 'torch_state_dicts'

PARAM_MAP = {"pm": "PM", "stator_yoke": "SY", "stator_tooth": "ST", "stator_winding": "SW"}
# sns.set_context("paper")
# plt.style.use("seaborn-whitegrid")


def show_error_trajectories(preds, dataset, sup_title=None):
    """4 columns for trends, error trend, input feats trend, and error histogram. Good vis for papers."""

    SUBSAMPLE_FACTOR = 1  # for smaller-file-sized pictures that look the same though
    target_feats = [c.replace('_pred', '') for c in preds if '_pred' in c]
    preds_df = preds.loc[:, [p for p in preds if '_pred' in p]].rename(
        columns={c: c.replace('_pred', '') for c in preds if '_pred' in c}) * dataset.temperature_scale
    gtruth_df = preds.loc[:, target_feats] * dataset.temperature_scale
    diff = preds_df - gtruth_df

    error_ratios = {}
    for c in target_feats:
        for thresh in (2.5, 5,):
            if thresh not in error_ratios:
                error_ratios[thresh] = dict()
            error_ratios[thresh][c] = len(diff.query(f'({c} > {thresh}) or ({c} < -{thresh})'))

    n_rows = 4  # pred vs. gtruth, error, input signals, histograms

    input_cols = dataset.input_cols[:len(dataset.target_cols)]
    pid_sizes = dataset.data.groupby(dataset.pid, sort=False).agg('size')
    test_set_l = preds.loc[:, dataset.pid].unique().tolist()

    vlines_x = np.cumsum(np.array([pid_sizes[int(pid)] for pid in test_set_l]))[:-1]
    min_y_abs = min(preds_df.min().min(), gtruth_df.min().min())
    max_y_abs = max(preds_df.max().max(), gtruth_df.max().max())
    min_y_err = diff.min().min()
    max_y_err = diff.max().max()
    # normalize
    inp_df = dataset.data.query(f"{dataset.pid} in @test_set_l").loc[:, input_cols].reset_index(drop=True)
    inp_df = (inp_df - inp_df.min(axis=0)) / (inp_df.max(axis=0) -
                                              inp_df.min(axis=0))

    # subsample
    if SUBSAMPLE_FACTOR > 1:
        inp_df = inp_df.iloc[::SUBSAMPLE_FACTOR, :]
        preds_df = preds_df.iloc[::SUBSAMPLE_FACTOR, :]
        gtruth_df = gtruth_df.iloc[::SUBSAMPLE_FACTOR, :]
        diff = diff.iloc[::SUBSAMPLE_FACTOR, :]

    annot_bbox_kws = {'facecolor': 'white', 'edgecolor': 'black',
                      'alpha': 0.3, 'pad': 1.0}
    vlines_kws = dict(colors='k', ls='dashed', zorder=3)

    fig, axes = plt.subplots(n_rows, len(target_feats), sharex=False, sharey='row',
                             figsize=(20, 3.3 * n_rows))

    for i, c in enumerate(gtruth_df):
        # plot signal measured and estimated
        # todo: Having only 1 target will break here
        #  axes is 1d then
        lbl = PARAM_MAP.get(c, c).replace("_", "-")
        ax = axes[0, i]
        ax.set_title(f'$\\vartheta_\\mathrm{{{lbl}}}$',
                     # fontdict=dict(fontsize=12)
                     )
        ax.plot(gtruth_df[c], color='lime', label='Ground truth', linestyle='-')
        ax.plot(preds_df[c], color='xkcd:indigo', label='Estimate', linestyle='-')
        ax.set_xlim(-1000, np.around(len(gtruth_df) * SUBSAMPLE_FACTOR, -3) + 300)
        ax.vlines(vlines_x, min_y_abs, max_y_abs, **vlines_kws)
        tcks = np.arange(0, np.around(len(gtruth_df)*SUBSAMPLE_FACTOR, -3), 10*3600)
        tcks_lbls = tcks // 3600
        if i == 0:
            ax.set_ylabel('Temperature in °C')
            ax_to_show_legend_from = ax

        ax.set_xticks(tcks)
        ax.set_xticklabels(tcks_lbls)
        #ax.set_ylim(None, 151)
        ax.grid(alpha=0.5)

        # plot signal estimation error
        ax = axes[1, i]
        ax.plot(diff[c], color='crimson',
                label='Temperature Estimation error ' +
                f'$\\vartheta_\\mathrm{{{lbl}}}$')
        ax.vlines(vlines_x, 2*min_y_err, 2*max_y_err, **vlines_kws)
        if i == 0:
            ax.set_ylabel('Error in °C')
        ax.set_ylim(-16, 16)
        ax.text(0.5, 1.03,
                s=f'MSE: {(diff[c] ** 2).mean():.2f} (°C)², ' + f'$||e||_\\infty$: {diff[c].abs().max():.2f} °C',
                #f'$||e||_\\infty\\uparrow$: {diff[c].max():.2f} °C, $||e||_\\infty\\downarrow$: {diff[c].min():.2f} °C',
                bbox=annot_bbox_kws,
                transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='center')
        ax.grid(alpha=0.5)
        ax.set_xticks(tcks)
        ax.set_xticklabels(tcks_lbls)
        # error histograms
        ax = axes[-1, i]
        ax.hist(diff[c], color='crimson', bins=100, density=True)
        if i == 0:
            ax.set_ylabel("Empirical probability")
        ax.set_xlabel('Estimation error in °C')

        ax.text(0.01, 0.9,
                s='\n'.join([f"Ratio over {thresh:>3.1f} °C: {errs_d[c] / len(diff):>3.3%}" for
                             thresh, errs_d in error_ratios.items()]),
                bbox=annot_bbox_kws,
                transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left'
                )

    for j, col in enumerate(input_cols):
        ax = axes[2, j]
        ax.plot(inp_df.loc[:, col], color='tab:grey',
                label=PARAM_MAP.get(col, col).replace("_", "-"))
        ax.set_xlabel('Time in hours')
        ax.set_title(PARAM_MAP.get(col, col).replace("_", "-"))
        if j == 0:
            ax.set_ylabel('Normalized qty')
        ax.grid(alpha=0.5)
        ax.vlines(vlines_x, 0, 1, **vlines_kws)
        ax.set_xticks(tcks)
        ax.set_xticklabels(tcks_lbls)
    if sup_title is not None:
        fig.suptitle(sup_title)
    fig.tight_layout()
    ax_to_show_legend_from.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0),
                                  # axes[0, 1].transAxes
                                  bbox_transform=fig.transFigure
                                  )


def show_error_trajectories_4_diss(preds, dataset, sup_title=None, large=True):
    """Error trajectories for dissertation. No Input features, but prediction vs gtruth, error trend, and error histogram.
    Expects preds to be unnormalized and pandas dataframes."""
    assert isinstance(preds, pd.DataFrame), "preds must be pandas DataFrame"
    test_set_l = preds.loc[:, dataset.pid].unique().tolist()
    gtruth_df = dataset.data.query(f"{dataset.pid} in @test_set_l").loc[:, dataset.target_cols].reset_index(drop=True)
    preds_df = preds.loc[:, dataset.target_cols].reset_index(drop=True)
    diff = preds_df - gtruth_df

    error_ratios = {}
    for c in dataset.target_cols:
        for thresh in (2.5, 5,):
            if thresh not in error_ratios:
                error_ratios[thresh] = dict()
            error_ratios[thresh][c] = len(diff.query(f'({c} > {thresh}) or ({c} < -{thresh})'))

    n_rows = 3  # pred vs. gtruth, error, histograms

    pid_sizes = dataset.get_pid_sizes()

    vlines_x = np.cumsum(np.array([pid_sizes[int(pid)] for pid in test_set_l]))[:-1]
    min_y_abs = min(preds_df.min().min(), gtruth_df.min().min())
    max_y_abs = max(preds_df.max().max(), gtruth_df.max().max())
    min_y_err = diff.min().min()
    max_y_err = diff.max().max()

    annot_bbox_kws = {'facecolor': 'white', 'edgecolor': 'black',
                      'alpha': 0.3, 'mutation_scale': 1.2}
    vlines_kws = dict(colors='k', ls='dashed', zorder=3)
    # for bigger pictures in jup notebooks with "talk" context even though context should be "paper"
    fig_scale = 1.0 # + int(large)*0.875
    fig_size = (6.49*fig_scale, 1.2*fig_scale*n_rows)
    fig, axes = plt.subplots(n_rows, len(dataset.target_cols), sharex="row", sharey='row',
                             figsize=fig_size, dpi=200)
    ERROR_LIM = 11  # in °C
    smaller_font_size = plt.rcParams['font.size']*0.8
    for i, c in enumerate(gtruth_df):
        # plot signal measured and estimated
        # todo: Having only 1 target will break here
        #  axes is 1d then
        lbl = PARAM_MAP.get(c, c).replace("_", "-")
        ax = axes[0, i]
        ax.set_title(f'$\\vartheta_\\mathrm{{{lbl}}}$')
        ax.plot(gtruth_df[c], color='lime', label='Ground truth', linestyle='-', lw=1)
        ax.plot(preds_df[c], color='xkcd:indigo', label='Estimate', linestyle='--', lw=1)
        ax.set_xlim(-1000, np.around(len(gtruth_df), -3) + 300)
        ax.vlines(vlines_x, 120, 140, **vlines_kws)
        tcks = np.arange(0, np.around(len(gtruth_df), -3), 3*7200)  # plot every 3 hours
        tcks_lbls = tcks // 7200
        if i == 0:
            ax.set_ylabel('Temperature in °C')
            ax_to_show_legend_from = ax

        ax.set_xticks(tcks)
        ax.set_xlabel("Time in h", labelpad=2.0)
        ax.set_xticklabels(tcks_lbls)  #  [])
        #ax.set_ylim(None, 151)
        ax.set_yticks(np.arange(20, 145, 40))
        prettify(ax)

        # plot signal estimation error
        ax = axes[1, i]
        ax.plot(diff[c], color='crimson', lw=1.0,
                label='Temperature Estimation error ' +
                f'$\\vartheta_\\mathrm{{{lbl}}}$')
        ax.vlines(vlines_x, 8, 10, **vlines_kws)
        if i == 0:
            ax.set_ylabel('Error in °C')
        ax.set_ylim(-ERROR_LIM, ERROR_LIM)

        ax.set_xticks(tcks)
        ax.set_xticklabels(tcks_lbls)
        ax.set_xlim(-1000, np.around(len(gtruth_df), -3) + 300)
        ax.set_xlabel("Time in h", labelpad=2.0)
        #ax.set_xlabel("Time in hours", labelpad=2.0)
        ax.set_yticks(np.arange(-10, 15, 5))
        prettify(ax)

        # error histograms
        ax = axes[-1, i]
        ax.hist(diff[c], color='crimson', bins=100, density=True)
        if i == 0:
            ax.set_ylabel("Empirical prob.")
        ax.set_xlabel('Error in °C', labelpad=2.0)
        ax.set_xlim(-ERROR_LIM, ERROR_LIM)
        ax.set_xticks(np.arange(-10, 15, 5))

        ax.text(0.05, 0.95,
                s='\n'.join([f"Ratio $>${thresh:>3.1f} °C: {100*errs_d[c] / len(diff):>3.1f}\\% " for
                             thresh, errs_d in error_ratios.items()]),
                #bbox=annot_bbox_kws,
                transform=ax.transAxes, fontsize=smaller_font_size,
                verticalalignment='top', horizontalalignment='left'
                )
        ax.text(0.1, -1.15,
                s=f'MSE: {(diff[c] ** 2).mean():.1f} (°C)²,\n$||e||_\\infty$: {diff[c].abs().max():.1f} °C',
                bbox=annot_bbox_kws,
                transform=ax.transAxes, fontsize=smaller_font_size,
                verticalalignment='bottom', horizontalalignment='left')
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(None, y_max*1.1)
        prettify(ax)

    if sup_title is not None:
        fig.suptitle(sup_title)
    # fig.tight_layout()
    ax_to_show_legend_from.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 0.92),
                                  # axes[0, 1].transAxes
                                  bbox_transform=fig.transFigure,
                                  frameon=False
                                  )
    fig.subplots_adjust(hspace=0.65, wspace=0.12)
    fig.align_ylabels(axes[:, 0])
    return fig


def main(expid, seed, debug=False):
    experiment_stats = DBM.query("SELECT * FROM trials as t "
                        "LEFT JOIN experiments as e ON t.experiment_id = e.id "
                        f"WHERE e.debug is false and e.id = '{expid}' "
                        "ORDER BY e.id ASC;").sort_values('mse')
    print(f"experiment stats has {len(experiment_stats)} entries")
    if seed is None:
        # take best seed
        trial_stats = experiment_stats.sort_values("mse", ascending=True).iloc[0, :]
        seed = trial_stats.loc["seed"]
    else:
        trial_stats = experiment_stats.query(f"seed == {seed}")
    pred_file = DBM.ESTIMATES_DUMP_PATH / f'expid_{expid}_{experiment_stats.model_tag.iloc[0]}_estimates.pkl.bz2'
    if not Path.exists(pred_file):
        host = experiment_stats.hostname.iloc[0]
        print(f"SecureCopy experiment {expid} from {host}")
        if host.startswith("n2cn"):
            # pc2
            remote_path = f"/scratch/hpc-prf-neunet/wilhelmk/temp_project_files/projects/mawk-2"\
                          f"/data/output/node_estimates/indexed_predictions/{pred_file.name}"
            host = 'noctua2-ln3'
        else:
            remote_path = pred_file
        subprocess.run(f"scp {host}:{remote_path} {pred_file}", shell=True)

    predictions = pd.read_pickle(pred_file).assign(exp_id=expid)

    # load initial data set
    ds = KaggleDataSet()
    ds.featurize()
    # ds.normalize()

    # filter predictions array
    loaded_preds = (predictions.groupby(['exp_id', 'repetition'])
                    .get_group((expid, seed))#.dropna('columns')
                    .drop(columns=['exp_id', 'repetition'])
                    .reset_index(drop=True))
    print(loaded_preds.head())
    loaded_preds.loc[:, ds.target_cols] *= ds.temperature_scale  # denormalize
    # add ground truth

    # plot test set performance

    show_error_trajectories_4_diss(loaded_preds, ds)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TNNs')
    parser.add_argument('-e', '--expid', default=None, required=True,
                        help='Experiment id to visualize')
    parser.add_argument('-s', '--seed', default=None, required=False,
                        help="Seed to visualize. Optional. Default is the best seed within experiment id")
    parser.add_argument("-d", "--debug", default=False, action="store_true",
                        help="Custom switch for experimental visuals.")
    args = parser.parse_args()
    main(int(args.expid), int(args.seed), args.debug)
