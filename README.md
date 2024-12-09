# diss-wk
Code, documentation, and figures for the dissertation of Wilhelm Kirchgässner.

The dissertation PDF is freely available at [UPB digital library](https://digital.ub.uni-paderborn.de/doi/10.17619/UNIPB/1-2068)

## Folder Structure and Code
```
.
├── LICENSE
├── notebooks
│   ├── 3.0-Ldiff_pmsm_diss_plot.ipynb
│   ├── 5.0-datasets_comp_diss_plots.ipynb
│   ├── 6.0-diss_linear_mdl_analysis.ipynb
│   ├── 6.1-diss_metrics.ipynb
│   ├── 6_7-diss_estimates_plots.ipynb
│   ├── 6_7-diss_hpo_plots.ipynb
│   ├── 7.1-diss_tnn_therm_props.ipynb
│   ├── 7.2-diss_lptn_analysis.ipynb
│   ├── 8.0-diss_conclusion.ipynb
│   └── img
├── README.md
├── requirements.txt
└── src
    ├── hpo_dyn_mdl_4_diss.py
    ├── hpo_lptn_4_diss.py
    ├── mawkutils
    │   ├── data.py
    │   ├── experiments.py
    │   ├── render.py
    │   ├── topology.py
    │   └── validation.py
    ├── run_blackbox_dyn_4_diss.py
    ├── run_blackbox_static_4_diss.py
    ├── run_custom_dyn_experiment_4_diss.py
    ├── run_custom_experiment_4_diss.py
    ├── run_linear_analysis_4_diss.py
    ├── run_lptn_4_diss.py
    ├── run_scipy_opt_lptn_4_diss.py
    └── visualize_performance.py
```

All code is split up in two parts: runner scripts under `src/` and visualizing jupyter notebooks under `notebooks/`.

The runner scripts have to be executed in order to train and cross-validate various machine learning models and to dump results into binaries and a database.

On the other hand, the jupyter notebooks merely visualize those dumped results, that is, no training and a minimum of inference is conducted there.

Runner scripts that start with `run_` will conduct a single experiment, whereas those that start with `hpo_` will conduct many experiments for a hyperparameter optimization.
Configuration for all experiments are usually in the runner scripts themselves.
Thus, in order to fill the database with all information expected by each cell in all notebooks will require several different configurations in the runner scripts.
Those should be evident by commented-out-code.

## Data 
The database is a local SQLITE DB, although the original research was conducted with a PostgreSQL DB on a server. 
The local DB is a means to enable reproducable code for fellow researchers and practitioners without access to a server with a PostgreSQL database management system.
Hence, plenty of code regarding the communication to a hypothetical server can be ignored by the common user.

Most of the code deals with one and the same data set, which is [available for free on Kaggle](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature).
The code expects this CSV to be located under `data/input/`.

The LPTN code partly needs additional data sets (two .mat files), that can be also found under `data/input/`.

All results from runner scripts will be stored under `data/output/`.

## Feedback / Troubleshooting
Please feel free to open up a GitHub issue for questions or concerns regarding this code base.
