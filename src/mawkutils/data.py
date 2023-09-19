"""This script provides helper functions and classes for File I/O and
database management"""

import psycopg2
import sshtunnel
from psycopg2 import connect as psql_connect
from psycopg2.extras import execute_batch
import sqlite3
import pathlib
import os
import platform
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from mawkutils.validation import calc_metrics


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class DBManager:
    """This class handles DB access. The dissertation was written in conjunction with experiment logging in form 
    of a PostgreSQL DB on a server in the department lab. For reproduction, the DB is dumped into a SQLITE DB,
    and future experiments will also be logged into a local SQLITE DB. This is to make it easer for fellow
    researchers to execute this code base where no server might be available."""
    SERVER_TAG = "lea38"  # server information is only applicable to a PostgreSQL DB
    SERVER_NAME = "lea-cyberdyne"  # server information is only applicable to a PostgreSQL DB
    PC2_LOCAL_PORT2PSQL = 12000  # server information is only applicable to a PostgreSQL DB
    HOSTS_WITH_DIRECT_PORT_ACCESS = (
        SERVER_NAME,
        "fe1",
        "ln-0001",
        "ln-0002",
    ) + tuple([f"n2login{i}" for i in range(100)])
    LEA_VPN_NODES = ("balamb", "altissia")  # server information is only applicable to a PostgreSQL DB
    TUNNEL_CFG = dict(remote_bind_address=("localhost", 5432))  # PSQL
    DBNAME = "mawk_node"
    MODEL_DUMP_PATH = (
        pathlib.Path.cwd().parent / "data" / "output" / "torch_state_dicts"
    )
    ESTIMATES_DUMP_PATH = (
        pathlib.Path.cwd().parent
        / "data"
        / "output"
        / "node_estimates"
        / "indexed_predictions"
    )
    TRENDS_DUMP_PATH = ESTIMATES_DUMP_PATH.parent / "indexed_trends"
    LOCAL_SQLITE_PATH = ESTIMATES_DUMP_PATH.parent / "table_dumps"

    # DB layout specifics (data types are relevant for SQLITE only)
    # For Postgresql, the DB was built upfront
    DB_LAYOUT = {
        "experiments": (
            ("model_tag", "text"),
            ("loss", "text"),
            ("cv", "text"),
            ("scriptname", "text"),
            ("hostname", "text"),
            ("debug", "integer"),
            ("dataset", "text"),
            ("n_folds", "integer"),
            ("input_cols", "text"),
            ("target_cols", "text"),
            ("model_size", "integer"),
            ("comment", "text"),
            ("layer_cfg", "text"),
        ),
        "trials": (
            ("experiment_id", "integer"),
            ("started_at", "text"),
            ("finished_at", "text"),
            ("seed", "integer"),
            ("mse", "real"),
            ("l_infty_over", "real"),
            ("l_infty_under", "real"),
        ),
    }
    SQLITE_FOREIGN_KEY_CONSTRAINTS = {
        "experiments": [],
        "trials": (
            "PRIMARY KEY (experiment_id, seed)",
            "FOREIGN KEY(experiment_id) REFERENCES experiments(id)",
        ),
    }

    def __init__(
        self,
        model_tag,
        loss_metric,
        cv,
        scriptname,
        model_arch,
        hostname,
        debugmode,
        dataset,
        n_folds,
        model_size,
        comment=None,
        save_trends=True,
        save_model=True,
    ) -> None:
        self.mdl_tag = model_tag
        self.scriptname = scriptname
        self.loss_metrics = loss_metric
        self.cv = cv
        self.hostname = hostname
        self.debugmode = debugmode
        self.n_folds = n_folds
        self.dataset = dataset
        self.model_size = model_size
        self.layer_cfg = model_arch
        self.comment = comment
        self.exp_id = (
            f'unknown_{"_".join(str(pd.Timestamp.now().round(freq="S")).split(" "))}'
        )
        self.do_save_trends = save_trends
        self.do_save_model = save_model
        pathlib.Path.mkdir(self.MODEL_DUMP_PATH, exist_ok=True, parents=True)
        pathlib.Path.mkdir(self.LOCAL_SQLITE_PATH, exist_ok=True, parents=True)
        pathlib.Path.mkdir(self.ESTIMATES_DUMP_PATH, exist_ok=True, parents=True)
        pathlib.Path.mkdir(self.TRENDS_DUMP_PATH, exist_ok=True, parents=True)

    @classmethod
    def conn_to_db_exists(cls):
        """Check whether connection to DB exists by checking if ssh tunnel can be opened.
        Skip if we are at server side or have direct port access (e.g., PC² frontend),
         or hostname is not among known hosts (which is directly a failed connection).
        """

        hostname = platform.uname().node
        conn_is_ok = False
        if hostname in cls.HOSTS_WITH_DIRECT_PORT_ACCESS:
            print(
                "Host is configured to have direct port access. No SSH tunnel checking."
            )
            conn_is_ok = True
        elif hostname in cls.LEA_VPN_NODES or hostname.lower().startswith(
            ("lea-", "cn-", "n2cn", "node")
        ):
            print("Test ssh tunnel..", end="", flush=True)
            if hostname.lower().startswith("cn-"):
                # noctua 1
                server_tag = "ln-0002"
                tun_cfg = dict(
                    remote_bind_address=("localhost", cls.PC2_LOCAL_PORT2PSQL),
                    ssh_username="wilhelmk",
                )
            elif hostname.lower().startswith("n2cn"):
                # noctua 2
                server_tag = "n2login3"
                tun_cfg = dict(
                    remote_bind_address=("localhost", cls.PC2_LOCAL_PORT2PSQL),
                    ssh_username="wilhelmk",
                )
            elif hostname.lower().startswith("node"):
                raise NotImplementedError(
                    "hostname 'node'. Where do you hang around? Oculus shouldnt exist anymore"
                )
            else:
                # lea workstation
                server_tag = cls.SERVER_TAG
                tun_cfg = cls.TUNNEL_CFG
            try:
                with sshtunnel.open_tunnel(server_tag, **tun_cfg) as tun:
                    conn_is_ok = tun.is_alive
            except Exception:
                pass
            print(
                f"{bcolors.OKGREEN}ok{bcolors.ENDC}"
                if conn_is_ok
                else f"{bcolors.FAIL}failed{bcolors.ENDC}"
            )
        else:
            print(
                "Unknown host. Assume there is no connection to server and "
                "fallback to local storage."
            )
        return conn_is_ok

    @staticmethod
    def read_credentials():
        """Read PSQL-DB credentials"""
        with open(Path(os.getenv("HOME")) / 'creds' / 'optuna_psql', "r") as f:
            cred_user, cred_pw = [
                s.strip() for s in f.readlines() if len(s.strip()) > 0
            ]
        return cred_user, cred_pw

    def save_log_to_psql_db(self, logs):
        """Save meta info to a database"""

        if self.conn_to_db_exists():
            cred_user, cred_pw = self.read_credentials()

            print(f"Save to DB..", end=" ")
            if self.hostname in self.HOSTS_WITH_DIRECT_PORT_ACCESS:
                # fe1 (=PC^2) frontend cannot create a tunnel,
                #  it has to be built upfront from server-side
                if self.hostname == self.SERVER_NAME:
                    host, port = self.TUNNEL_CFG["remote_bind_address"]
                else:
                    host, port = "localhost", self.PC2_LOCAL_PORT2PSQL
                with psql_connect(
                    dbname=self.DBNAME,
                    user=cred_user,
                    password=cred_pw,
                    host=host,
                    port=port,
                ) as conn:
                    self.exp_id = self.save_over_connection(conn, logs)
                conn.close()
            else:
                tun_cfg = self.TUNNEL_CFG.copy()
                if (
                    self.hostname in self.LEA_VPN_NODES
                    or self.hostname.lower().startswith("lea-")
                ):
                    # we are in LEA VPN network
                    server_name = self.SERVER_TAG
                else:
                    # assume we are on a PC2 compute node
                    assert self.hostname.lower().startswith(
                        ("cn-", "n2cn")
                    ), f"host {self.hostname} is not a PC² compute node"
                    server_name = (
                        "ln-0002"
                        if self.hostname.lower().startswith("cn-")
                        else "n2login3"
                    )
                    tun_cfg["ssh_username"] = os.getenv('USER')  # assume the same user name on client and server
                    tun_cfg["remote_bind_address"] = (
                        "localhost",
                        self.PC2_LOCAL_PORT2PSQL,
                    )
                with sshtunnel.open_tunnel(server_name, **tun_cfg) as tun:
                    with psql_connect(
                        dbname=self.DBNAME,
                        user=cred_user,
                        password=cred_pw,
                        host=tun.local_bind_host,
                        port=tun.local_bind_port,
                    ) as conn:
                        self.exp_id = self.save_over_connection(conn, logs)
                    conn.close()

        else:
            print("No connection to server - Saving logs/meta info to SQLite.")
            with sqlite3.connect(self.LOCAL_SQLITE_PATH / "mawk_node.sqlite") as conn:
                # init if not exists
                for table, layout in self.DB_LAYOUT.items():
                    if table == "experiments":
                        layout = [("id", "integer primary key")] + list(layout)
                    layout = [f"{n} {t}" for n, t in layout]
                    layout += self.SQLITE_FOREIGN_KEY_CONSTRAINTS[table]
                    query = f"CREATE TABLE IF NOT EXISTS {table}({', '.join(layout)});"
                    conn.execute(query)
                self.exp_id = self.save_over_connection(conn, logs)
            conn.close()  # apparently necessary despite context-manager

    def save_over_connection(self, conn, logs, engine="psql"):
        """Execute INSERT queries.

        Args:
            conn : PSQL or SQLITE connection
            logs (list of dicts): Data to insert.
            engine (str, optional): DB type. Either "psql" or "sqlite". Defaults to "psql".

        Returns:
            [int]: The newly generated experiment id.
        """

        if isinstance(conn, sqlite3.Connection):
            engine = "sqlite"
        elif isinstance(conn, psycopg2.extensions.connection):
            engine = "psql"
        else:
            raise ValueError(f"engine not recognizable from connection: {conn}")
        placeholder = "%s" if engine == "psql" else "?"
        exp_query = (
            f"INSERT INTO experiments ({', '.join([i for i,_ in self.DB_LAYOUT['experiments']])}) "
            f"VALUES ({', '.join(len(self.DB_LAYOUT['experiments'])*[placeholder])}) RETURNING id;"
        )
        trials_query = (
            f"INSERT INTO trials ({', '.join([i for i,_ in self.DB_LAYOUT['trials']])}) "
            f"VALUES ({', '.join(len(self.DB_LAYOUT['trials'])*[placeholder])});"
        )

        def maybe_convert_for_sqlite(entry):
            if engine == "sqlite":
                entry = [
                    str(e) if isinstance(e, (list, pd.Timestamp)) else e for e in entry
                ]
            return entry

        def get_experiments_tup():
            """Return attributes according to experiments table layout"""
            ret = (
                self.mdl_tag,
                self.loss_metrics,
                self.cv,
                self.scriptname,
                self.hostname,
                self.debugmode,
                self.dataset.name,
                self.n_folds,
                self.dataset.input_cols,
                self.dataset.target_cols,
                self.model_size,
                self.comment,
                self.layer_cfg,
            )
            ret = maybe_convert_for_sqlite(ret)
            return ret

        sane_logs = [
            l for l in logs if not l["all_predictions_df"].isnull().values.any()
        ]
        if len(sane_logs) != len(logs):
            warnings.warn(
                f"{len(logs) - len(sane_logs)} of {len(logs)} experiments ended up with NaNs"
            )
        if len(sane_logs) == 0:
            warnings.warn("All experiments failed. Return 999 for all.")
            metrics = [dict(mse=999, l_infty_over=999, l_infty_under=999) for _ in logs]
        else:
            metrics = [
                calc_metrics(
                    l["all_predictions_df"].loc[:, self.dataset.target_cols],
                    l["ground_truth"].loc[:, self.dataset.target_cols],
                    target_scale=self.dataset.temperature_scale,
                )
                for l in sane_logs
            ]

        def get_trials_tup(e_id):
            """Return attributes according to trials table layout"""
            ret = [
                (
                    e_id,
                    l["start_time"],
                    l["end_time"],
                    l["seed"],
                    m["mse"],
                    m["l_infty_over"],
                    m["l_infty_under"],
                )
                for l, m in zip(sane_logs, metrics)
            ]

            ret = [maybe_convert_for_sqlite(l) for l in ret]
            return ret

        # execute query
        if engine == "psql":
            with conn.cursor() as curs:
                curs.execute(exp_query, get_experiments_tup())
                exp_id = curs.fetchone()[0]
                execute_batch(curs, trials_query, get_trials_tup(exp_id))
        else:  # sqlite
            curs = conn.cursor()  # no context-manager available
            curs.execute(exp_query, get_experiments_tup())
            exp_id = curs.fetchone()[0]
            curs.executemany(trials_query, get_trials_tup(exp_id))
            curs.close()
        print(f"inserted exp_id: {bcolors.OKGREEN}{exp_id}{bcolors.ENDC}")

        # maybe dump model weights
        if self.do_save_model:
            for log in sane_logs:
                for fold_i, fold_model in enumerate(log["models_state_dict"]):
                    fname = (
                        "dummy_model.pt"
                        if self.debugmode
                        else f"exp_{exp_id}_seed_{log['seed']}_fold_{fold_i}.pt"
                    )
                    torch.save(fold_model, self.MODEL_DUMP_PATH / fname)
        return exp_id

    def save_predictions_to_disk(self, logs):
        print("Save predictions..", end="", flush=True)
        if self.debugmode:
            fname = "dummy_debug_estimates.pkl.bz2"
        else:
            fname = f"expid_{self.exp_id}_{self.mdl_tag}_estimates.pkl.bz2"
        all_preds = pd.concat(
            [log["all_predictions_df"] for log in logs], ignore_index=True
        )
        try:
            pathlib.Path.mkdir(self.ESTIMATES_DUMP_PATH, exist_ok=True, parents=True)
            all_preds.to_pickle(self.ESTIMATES_DUMP_PATH / fname)
            print(f"{bcolors.OKGREEN}ok{bcolors.ENDC}")
        except Exception:
            print(f"{bcolors.FAIL}failed{bcolors.ENDC}")

    def save_trends(self, logs):
        print("Save trends..", end="", flush=True)

        trend_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "training": train_arr,
                        "validation": val_arr if len(val_arr) > 0 else np.nan,
                        "fold": fold_i,
                        "rep": log_i,
                    }
                )
                for log_i, log in enumerate(logs)
                for fold_i, (train_arr, val_arr) in enumerate(
                    zip(log["loss_trends_train"], log["loss_trends_val"])
                )
            ],
            ignore_index=True,
        )
        if self.debugmode:
            fname = "dummy_debug_trends.pkl.bz2"
        else:
            fname = f"expid_{self.exp_id}_{self.mdl_tag}_trends.pkl.bz2"
        try:
            trend_df.to_pickle(self.TRENDS_DUMP_PATH / fname)
            print(f"{bcolors.OKGREEN}ok{bcolors.ENDC}")
        except Exception:
            print(f"{bcolors.FAIL}failed{bcolors.ENDC}")

    def save(self, logs):
        self.save_log_to_psql_db(logs)
        self.save_predictions_to_disk(logs)
        if self.do_save_trends:
            self.save_trends(logs)

    @classmethod
    def query(
        cls,
        db_query,
    ):
        """Query from DB (remotely) or, if no connection is available, query from
        local SQLITE DB.

        Args:
            db_query (str): The query to send to the PSQL (or SQLITE) DBMS.
        """

        if cls.conn_to_db_exists():
            # read data from DB
            #  we assume we are neither on the server comprising the DB,
            #  nor on the PC²-frontend where a tunnel is prebuilt,
            #  nor on a PC² compute node, where a tunnel to the frontend would have to be built
            with sshtunnel.open_tunnel(cls.SERVER_TAG, **cls.TUNNEL_CFG) as tun:
                user, pw = cls.read_credentials()
                with psql_connect(
                    dbname=cls.DBNAME,
                    user=user,
                    password=pw,
                    host=tun.local_bind_host,
                    port=tun.local_bind_port,
                ) as conn:
                    df = pd.read_sql_query(db_query, conn)
        else:
            # read DB dumps locally (sqlite)
            with sqlite3.connect(cls.LOCAL_SQLITE_PATH / "mawk_node.sqlite") as conn:
                df = pd.read_sql_query(db_query, conn)
            conn.close()  # necessary according to docs
            # sanitize list entries
            df = df.assign(
                **{
                    lc: df.loc[:, lc].transform(lambda x: x.strip("{}").split(","))
                    for lc in ["target_cols", "input_cols"]
                }
            )
        conn.close()
        return df


class DataSet:
    input_cols = []
    target_cols = []
    extra_cols = []
    dataset_path = None
    input_temperature_cols = []
    pid = "Not_Available"
    temperature_scale = 200  # in °C

    def __init__(self, input_cols=None, target_cols=None, with_extra_cols=False):
        # make it possible to load more input cols than supposed by class definition
        input_cols = input_cols or self.input_cols
        target_cols = target_cols or self.target_cols
        if self.dataset_path.suffixes[0] == ".csv":
            self.data = pd.read_csv(self.dataset_path)
        else:
            # assume pickled file
            self.data = pd.read_pickle(self.dataset_path)
        col_arrangement = input_cols + [self.pid] + target_cols
        if with_extra_cols:
            col_arrangement += self.extra_cols
        self.data = self.data.loc[:, [c for c in col_arrangement if c in self.data]]
        # note, some features in input/target cols will only exist after featurizing!
        self.input_cols = [c for c in input_cols if c in self.data]
        self.target_cols = [c for c in target_cols if c in self.data]
        self.black_list = []  # some columns need to be dropped after featurizing

    @property
    def temperature_cols(self):
        return self.target_cols + self.input_temperature_cols

    @property
    def non_temperature_cols(self):
        return [
            c
            for c in self.data
            if c not in self.temperature_cols + [self.pid, "train_" + self.pid]
        ]

    def get_pid_sizes(self, pid_lbl=None):
        """Returns pid size as pandas Series"""
        pid_lbl = pid_lbl or self.pid
        return self.data.groupby(pid_lbl).agg("size").sort_values(ascending=False)

    def normalize(self):
        """Simple division by a scale, no offsets"""
        # Be wary that changing target_cols in featurize()
        #  and calling it after this normalize function will bring unexpected behavior
        #  e.g., adding a temperature to target cols in featurize and calling it after normalize will have
        #   that new target temperature normalized on its max value instead of temp_denom
        nt_cols = [c for c in self.non_temperature_cols if c in self.data]
        t_cols = [c for c in self.temperature_cols if c in self.data]
        # some columns might only exist after featurize()
        self.data.loc[:, t_cols] /= self.temperature_scale
        self.data.loc[:, nt_cols] /= self.data.loc[:, nt_cols].abs().max(axis=0)

    def get_profiles_for_cv(self, cv_lbl, kfold_split=4):
        """Given a cross-validation label and a table of profile sizes, return a tuple which associates
        training, validation and test sets with profile IDs.

        Args:
            cv_lbl (str): Cross-validation label. Allowed labels can be seen in wkutils.config.
            kfold_split (int, optional): The number of profiles per fold. Only active if cv_lbl=='kfold'. Defaults to 4.

        Returns:
            Tuple: training, validation and test set lists of lists of profile IDs fanned out by fold.
        """
        raise NotImplementedError()


class KaggleDataSet(DataSet):
    input_cols = ["ambient", "coolant", "u_d", "u_q", "motor_speed", "i_d", "i_q"]
    target_cols = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    input_temperature_cols = ["ambient", "coolant"]
    extra_cols = ["torque"]
    dataset_path = Path().cwd().parent / "data" / "input" / "measures_v2.csv"
    pid = "profile_id"
    name = "kaggle"
    sample_time = 0.5  # in seconds

    def get_profiles_for_cv(self, cv_lbl, verbose=True):
        """Returns a 3-elem-tuple for train, val, test set, each being a list of profile ids."""
        pid_sizes = self.get_pid_sizes()
        black_listed_profiles = []
        if isinstance(cv_lbl, str):
            if cv_lbl == "1fold":
                test_profiles = [[60, 62, 74]]
                validation_profiles = [[4]]
            elif cv_lbl == "1fold_no_val":
                test_profiles = [[60, 62, 74]]
                validation_profiles = [[]]
            elif cv_lbl == "1fold_static_diss":
                test_profiles = [[16, 20, 48, 53, 60]]
                validation_profiles = [[71]]
            elif cv_lbl == "1fold_no_val_static_diss":
                test_profiles = [[16, 20, 48, 53, 60]]
                validation_profiles = [[]]
            elif cv_lbl == "hpo_1fold_no_val_static_diss":
                black_listed_profiles = [16, 20, 48, 53, 60]  # gen set
                test_profiles = [[65]]
                validation_profiles = [[]]
            elif cv_lbl == "hpo_1fold_diss":
                black_listed_profiles = [16, 20, 48, 53, 60]  # gen set
                test_profiles = [[65]]
                validation_profiles = [[71]]
            else:
                raise NotImplementedError(f"cv '{cv_lbl}' is not implemented.")
        elif callable(cv_lbl):
            validation_profiles, test_profiles, black_listed_profiles = cv_lbl()
        train_profiles = [
            [
                p
                for p in pid_sizes.index.tolist()
                if p not in tst_p + val_p + black_listed_profiles
            ]
            for tst_p, val_p in zip(test_profiles, validation_profiles)
        ]
        if verbose:
            # print size of each fold
            for profile in test_profiles:
                test_set_size = pid_sizes.loc[profile].sum()
                print(
                    f"Fold {0} test size: {test_set_size} samples ({test_set_size / pid_sizes.sum():.1%} of total)"
                )

        return train_profiles, validation_profiles, test_profiles

    def featurize(self, scheme="basic"):
        # extra feats (FE)
        # it is highly advisable to call featurize and then normalize, not the other way around!
        # Because featurize might mess with input and target cols
        if {"i_d", "i_q", "u_d", "u_q"}.issubset(set(self.data.columns.tolist())):
            if scheme == "basic":
                extra_feats = {
                    "i_s": lambda x: np.sqrt((x["i_d"] ** 2 + x["i_q"] ** 2)),
                    "u_s": lambda x: np.sqrt((x["u_d"] ** 2 + x["u_q"] ** 2)),
                }
            elif scheme == "extensive":
                extra_feats = {
                    "i_s": lambda x: np.sqrt((x["i_d"] ** 2 + x["i_q"] ** 2)),
                    "u_s": lambda x: np.sqrt((x["u_d"] ** 2 + x["u_q"] ** 2)),
                    "S_el": lambda x: x["i_s"] * x["u_s"],
                    "P_el": lambda x: x["i_d"] * x["u_d"] + x["i_q"] * x["u_q"],
                    "i_s_x_w": lambda x: x["i_s"] * x["motor_speed"],
                    "S_x_w": lambda x: x["S_el"] * x["motor_speed"],
                }
            elif scheme == "plain":
                extra_feats = {}
        self.data = self.data.assign(**extra_feats).drop(columns=self.black_list)
        self.input_cols = [
            c for c in self.data if c not in self.target_cols + [self.pid]
        ]
        # rearrange
        self.data = self.data.loc[:, self.input_cols + [self.pid] + self.target_cols]


class ChunkedKaggleDataSet(KaggleDataSet):
    name = "chunked_kaggle"

    def __init__(
        self,
        input_cols=None,
        target_cols=None,
        chunk_size=None,
        cv="1fold",
        with_extra_cols=False,
    ):
        super().__init__(
            input_cols=input_cols,
            target_cols=target_cols,
            with_extra_cols=with_extra_cols,
        )
        p_len = 1  # in hours
        chunk_size = chunk_size or int(p_len * 3600 / self.sample_time)
        tra_l, val_l, tst_l = self.get_profiles_for_cv(cv_lbl=cv)
        assert len(tra_l) == 1 and len(val_l) == 1 and len(tst_l) == 1, (
            "Chunked Kaggle data set with a k-fold CV and k > 1 is not implemented. "
            f"{len(tra_l)=},{len(val_l)=},{len(tst_l)=}"
        )
        tra_l, val_l, tst_l = tra_l[0], val_l[0], tst_l[0]
        tmp_profiles = []
        for pid, df in self.data.groupby(self.pid, sort=False):
            if pid in val_l + tst_l:
                tmp_profiles.append(df)
            elif pid in tra_l:
                tmp_profiles.extend(
                    [
                        df.iloc[n : min(n + chunk_size, len(df)), :].assign(
                            **{self.pid: pid + i * 1000}
                        )
                        for i, n in enumerate(range(0, len(df), chunk_size), start=1)
                    ]
                )
            else:
                # black-listed profiles (e.g., for HPO)
                pass

        self.data = pd.concat(tmp_profiles, ignore_index=True)  # flatten


class SmoothKaggleDataSet(KaggleDataSet):
    dataset_path = Path().cwd().parent / "data" / "input" / "smooth_measures_v2.csv"


class ChunkedSmoothKaggleDataSet(ChunkedKaggleDataSet):
    dataset_path = Path().cwd().parent / "data" / "input" / "smooth_measures_v2.csv"


EWM_ADJUST = False  # better


def add_mov_stats(data_1, input_feats, pid, spans=None, add_ewma=True, add_ewms=False):
    ewma_feats_df_l = []
    spans = spans or [1320, 3360, 6360, 9480]
    for p_id, df in data_1.groupby(pid, sort=False):
        df = df.reset_index(drop=True)
        x = df.loc[:, input_feats]
        ew_l = []
        if add_ewma:
            ew_l += [
                x.ewm(span=s, adjust=EWM_ADJUST)
                .mean()
                .rename(columns=lambda c: f"{c}_ewma_{s}")
                for s in spans
            ]
        if add_ewms:
            ew_l += [
                x.ewm(span=s, adjust=EWM_ADJUST)
                .std()
                .rename(columns=lambda c: f"{c}_ewms_{s}")
                .fillna(0)
                for s in spans
            ]
        ewma_feats_df_l.append(pd.concat([df] + ew_l, axis=1))
    return pd.concat(ewma_feats_df_l, ignore_index=True)
