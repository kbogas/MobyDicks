import time

import cached_path
import lightning as L
import numpy as np
import pandas as pd
import torch as th
from graph_model import MLP, GraphAny, generate_logit_dicts, get_metrics_from_torch
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from networkx import non_neighbors
from pmlb import fetch_data
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sympy import re
from torch import optim, rand
from torch.utils.data import DataLoader, TensorDataset


class Lightning_GraphAny(L.LightningModule):
    def __init__(
        self,
        n_hidden,
        feat_channels,
        pred_channels,
        att_temperature,
        entropy=1,
        n_mlp_layer=2,
        device="cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.graphany = GraphAny(
            n_hidden,
            feat_channels,
            pred_channels,
            att_temperature,
            entropy=1,
            n_mlp_layer=2,
            device=device,
            **kwargs,
        )

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        # if input_type[0].item() == 0:
        #     logit_dicts = logit_dicts_train
        # elif input_type[0].item() == 1:
        #     logit_dicts = logit_dicts_test
        # elif input_type[0].item() == 2:
        #     logit_dicts = logit_dicts_val
        # else:
        #     raise NotImplementedError(f'Batch size is currently used to differentiate between train/val/test (64/16/32). Do not change {batch_indices.shape[0]}')
        y_pred, att = self.graphany(batch_indices, logit_dicts_train)
        loss = th.nn.functional.mse_loss(y, y_pred)

        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)

        values = {"train_loss": loss, "train_f1": f1}  # add more items if needed
        self.log_dict(values, prog_bar=True)

        return loss

    def validation_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        y_pred, att = self.graphany(batch_indices, logit_dicts_val)
        loss = th.nn.functional.mse_loss(y, y_pred)
        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)
        self.log_dict({"val_loss": loss, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        y_pred, att = self.graphany(batch_indices, logit_dicts_test)
        loss = th.nn.functional.mse_loss(y, y_pred)
        return loss

    def predict_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        y_pred, att = self.graphany(batch_indices, logit_dicts_test)
        return y_pred

    def configure_optimizers(self):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.02},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # AdamW
        optimizer = th.optim.AdamW(
            optim_groups,
            lr=1e-3,
            weight_decay=0.02,
        )
        return optimizer


def train_graphany(
    logit_dicts_train,
    logit_dicts_val,
    logit_dicts_test,
    feat_channels=["linear", "h_1", "l_1"],
):

    device = "cuda:0"
    model = Lightning_GraphAny(
        n_hidden=128,
        feat_channels=feat_channels,
        pred_channels=feat_channels,
        att_temperature=5,
        entropy=1,
        mlp_layer=2,
        device=device,
    )

    trainer = L.Trainer(
        deterministic=True,
        devices=[0],
        accelerator="gpu",
        max_epochs=500,
        # enable_model_summary=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001),
            RichProgressBar(leave=True),
        ],
        check_val_every_n_epoch=5,
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer, model


class Lightning_MLP(L.LightningModule):
    def __init__(self, num_feats, total_budget, device="cuda:0", **kwargs):
        super().__init__()

        self.mlp = construct_mlp(num_feats, total_budget, num_class=2)
        self.mlp.to(self.device)

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch
        y = y.type(th.float32)

        # if input_type[0].item() == 0:
        #     logit_dicts = logit_dicts_train
        # elif input_type[0].item() == 1:
        #     logit_dicts = logit_dicts_test
        # elif input_type[0].item() == 2:
        #     logit_dicts = logit_dicts_val
        # else:
        #     raise NotImplementedError(f'Batch size is currently used to differentiate between train/val/test (64/16/32). Do not change {batch_indices.shape[0]}')
        data = th.tensor(X_train[batch_indices.cpu().numpy()], dtype=th.float32).to(
            self.device
        )
        y_pred = self.mlp(data)
        loss = th.nn.functional.mse_loss(y, y_pred)

        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)

        values = {"train_loss": loss, "train_f1": f1}  # add more items if needed
        self.log_dict(values, prog_bar=True)

        return loss

    def validation_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        data = th.tensor(X_val[batch_indices.cpu().numpy()], dtype=th.float32).to(
            self.device
        )
        y_pred = self.mlp(data)
        loss = th.nn.functional.mse_loss(y, y_pred)
        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)
        self.log_dict({"val_loss": loss, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        data = th.tensor(X_test[batch_indices.cpu().numpy()], dtype=th.float32).to(
            self.device
        )
        y_pred = self.mlp(data)
        loss = th.nn.functional.mse_loss(y, y_pred)
        return loss

    def predict_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        batch_indices, y = batch

        data = th.tensor(X_test[batch_indices.cpu().numpy()], dtype=th.float32).to(
            self.device
        )
        y_pred = self.mlp(data)
        return y_pred

    def configure_optimizers(self):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.02},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # AdamW
        optimizer = th.optim.AdamW(
            optim_groups,
            lr=1e-3,
            weight_decay=0.02,
        )
        return optimizer


def train_mlp(num_feats, total_budget):

    device = "cuda:0"
    model = Lightning_MLP(num_feats=num_feats, total_budget=total_budget, device=device)
    trainer = L.Trainer(
        deterministic=True,
        devices=[0],
        accelerator="gpu",
        max_epochs=500,
        # enable_model_summary=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001),
            RichProgressBar(leave=True),
        ],
        check_val_every_n_epoch=5,
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer, model


def generate_pred(trainer, model, test_dataloader):
    predictions = trainer.predict(model=model, dataloaders=test_dataloader)
    predictions = th.vstack(predictions)
    y_pred = predictions.argmax(dim=1).detach().numpy()
    return y_pred


def construct_mlp(num_feats, budget, num_class=2):
    num_hidden = int(np.round(np.roots([1, num_class + num_feats, -budget]).max(), 0))
    mlp = MLP(
        in_channels=num_feats,
        hidden_channels=num_hidden,
        out_channels=num_class,
        n_layers=2,
    )
    return mlp


def set_seeds(seed=42):
    np.random.seed(seed)
    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=True)


def get_edge_homophily(A, y):
    y_ = y.copy()
    y_[y_ == 0] = -1
    edge_homophily = ((A @ y > 0) == (y > 0)).sum() / len(y)
    return edge_homophily


random_state = 42

number_of_cv_folds = 5

cv = StratifiedKFold(number_of_cv_folds, random_state=random_state, shuffle=True)
path_to_csv = "./neq_diff.csv"

# models = {
#     "DT": {},
#     "RF": {},
#     "LR": {},
#     "kNN": {},
#     "GraphAny_trans": {},
#     "GraphAny_linear": {'feat_channels':['linear']},
#     "GraphAny_ind": {},
#     "NEQ": {},
#     "GraphAny_mlp": {},
# }


from scipy.special import softmax
from sklearn.base import BaseEstimator


class NEQBoost(BaseEstimator):

    def __init__(self, strategy="min_residual_selection"):
        self.ohe = OneHotEncoder()
        self.sc = StandardScaler()
        self.strategy = strategy
        self.W = []

    def fit(self, X, y):

        X_leftover = self.sc.fit_transform(X)
        y_leftover = self.ohe.fit_transform(y.reshape(-1, 1)).toarray()
        wrong = np.arange(X_leftover.shape[0])
        while sum(wrong) > 0:
            X_leftover, y_leftover = X_leftover[wrong], y_leftover[wrong]
            W_leftover, _, _, _ = np.linalg.lstsq(X_leftover, y_leftover, rcond=None)
            preds_ohe = X_leftover @ W_leftover
            wrong = preds_ohe.argmax(axis=1) != y_leftover.argmax(axis=1)
            self.W.append(W_leftover)
            # print(sum(wrong))
        self.num_neqs = len(self.W)
        # print(f"Rounds of boosting: {self.num_neqs}")
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)

    def predict_proba(self, X):
        X_sc = self.sc.transform(X)
        y_probas_all = []
        dists = []
        for W in self.W:
            y_logit = X_sc @ W
            dist = np.linalg.norm(y_logit, axis=1)
            y_proba = softmax(y_logit, axis=1)
            y_probas_all.append(y_proba)
            dists.append(dist)
        dists = np.vstack(dists).T
        dists = softmax(dists, axis=1)
        y_probas_all = np.array(y_probas_all).reshape(self.num_neqs, X.shape[0], -1)
        if self.strategy == "weighted_residual_voting":
            y_probas = np.einsum("mic, im-> ic", y_probas_all, dists)
        elif self.strategy == "min_residual_selection":
            neq_to_use = dists.argmin(axis=1)
            y_probas = y_probas_all[neq_to_use, np.arange(X.shape[0]), :]
        else:
            raise NotImplementedError(f"Can't understand {self.strategy}")
        return y_probas


from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import BallTree, KNeighborsClassifier


def fit_subsamples(X_leftovers, labels_leftovers, max_to_check="sqrt"):
    if max_to_check == "sqrt":
        max_num_clusters = int(np.round(np.sqrt(X_leftovers.shape[0]), 0)) + 1
    elif max_to_check == "full":
        max_num_clusters = X_leftovers.shape[0] + 1
    elif isinstance(max_to_check, int):
        max_num_clusters = max_to_check
    else:
        raise NotImplementedError()

    res = []
    cluster_dict = {}
    for k in range(1, max_num_clusters):
        cl = KMeans(n_clusters=k, n_init="auto")
        cl.fit(X_leftovers)
        cluster_dict[k] = {"cl": cl, "W": []}
        rr = 0
        acc = 0
        for unq_label in np.unique(cl.labels_):
            current_cluster = np.where(cl.labels_ == unq_label)[0]
            X_cur, y_cur_ohe = (
                X_leftovers[current_cluster],
                labels_leftovers[current_cluster],
            )
            W_ohe_cur, residual_sums, _, _ = np.linalg.lstsq(
                X_cur, y_cur_ohe, rcond=None
            )
            cluster_dict[k]["W"].append(W_ohe_cur)
            cur_acc = accuracy_score(
                y_cur_ohe.argmax(axis=1), (X_cur @ W_ohe_cur).argmax(axis=1)
            )
            rr += residual_sums.sum()
            acc += cur_acc

        # print(rr)
        res.append((int(k), cl.inertia_, rr, acc / k))
        # break
    res_df = pd.DataFrame(res, columns=["k", "inertia", "ss", "acc"])
    wanted_k = res_df.sort_values(["acc", "k"], ascending=[False, True]).iloc[0]["k"]
    return cluster_dict[wanted_k]


class NEQ_Local(BaseEstimator):

    def __init__(self, strategy="selection", k="sqrt"):
        self.ohe = OneHotEncoder()
        self.sc = StandardScaler()
        self.k = k
        if isinstance(self.k, int):
            self.num_neigh = self.k
        self.strategy = strategy
        self.W = None
        self.subsamples_dict = {}
        self.neighbor_tree = None

    def fit(self, X, y):

        X_sc = self.sc.fit_transform(X)

        if self.k == "auto":
            accs = []
            for k in range(1, int(np.sqrt(X.shape[0]))):
                kn = KNeighborsClassifier(n_neighbors=k)
                kn.fit(X_sc, y)
                accs.append(kn.score(X_sc, y))
            self.num_neigh = np.argmax(accs) + 1

        elif self.k == "sqrt":
            self.num_neigh = int(np.sqrt(X.shape[0]))

        self.num_neigh = (
            self.num_neigh + 1 if self.num_neigh % 2 == 0 else self.num_neigh
        )

        self.neighbor_tree = BallTree(X_sc)

        y_ohe = self.ohe.fit_transform(y.reshape(-1, 1)).toarray()

        self.W, _, _, _ = np.linalg.lstsq(X_sc, y_ohe, rcond=None)
        preds_ohe = X_sc @ self.W
        correct = (preds_ohe.argmax(axis=1) == y).astype(int)
        self.neq_train_labels = correct
        if (~correct.astype(bool)).sum() > 0:
            X_leftovers = X_sc[~correct.astype(bool)]
            labels_leftovers = y_ohe[~correct.astype(bool)]
            self.subsamples_dict = fit_subsamples(X_leftovers, labels_leftovers)
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)

    def predict_proba(self, X):
        X_sc = self.sc.transform(X)
        dist, indices = self.neighbor_tree.query(X_sc, k=self.num_neigh)
        # num_test X 1
        neq_percentage_correct = self.neq_train_labels[indices].mean(axis=1)
        y_neq = X_sc @ self.W
        if len(self.subsamples_dict) > 0:

            if self.strategy == "selection":
                keep_neq = neq_percentage_correct > 0.5
                if (~keep_neq).sum():
                    X_leftovers = X_sc[~keep_neq]
                    local_neq_to_use = self.subsamples_dict["cl"].predict(X_leftovers)
                    y_locals = []
                    for sample_index, neq_index in enumerate(local_neq_to_use):
                        cur_X = X_leftovers[sample_index, :]
                        cur_W = self.subsamples_dict["W"][neq_index]
                        y_locals.append(cur_X @ cur_W)
                    y_locals = np.vstack(y_locals)
                    y_neq[~keep_neq] = y_locals
            elif self.strategy == "weighted_voting":
                keep_neq = neq_percentage_correct > 0.5
                if (~keep_neq).sum():
                    X_leftovers = X_sc[~keep_neq]
                    perc_nec = neq_percentage_correct[~keep_neq]
                    local_neq_to_use = self.subsamples_dict["cl"].predict(X_leftovers)
                    y_locals = []
                    for sample_index, neq_index in enumerate(local_neq_to_use):
                        cur_X = X_leftovers[sample_index, :]
                        cur_W = self.subsamples_dict["W"][neq_index]
                        y_locals.append(cur_X @ cur_W)
                    y_locals = np.vstack(y_locals)
                    y_neq[~keep_neq] = perc_nec.reshape(-1, 1) * softmax(
                        y_neq[~keep_neq], axis=1
                    ) + (1 - perc_nec).reshape(-1, 1) * softmax(y_locals, axis=1)
            else:
                raise NotImplementedError(f"Can't understand {self.strategy}")
        y_probas = softmax(y_neq, axis=1)
        return y_probas


models = {
    # 'NEQ_Local_weighted_auto': {'strategy':'weighted_voting', 'k':'auto'},
    # 'NEQ_Local_selection_sqrt': {'strategy':'selection', 'k':'sqrt'},
    # 'NEQ_Local_weighted_sqrt': {'strategy':'weighted_voting', 'k':'sqrt'},
    # 'NEQ_Local_selection_auto': {'strategy':'selection', 'k':'auto'},
    # 'NEQ_Local_selection_5': {'strategy':'selection', 'k':5},
    #'Mean': {'strategy': 'mean', 'names': ['straight', 'l1', 'h1']},
    #'Mean_2': {'strategy': 'mean', 'names': ['straight']}
    "Boost_min": {"strategy": "min_residual_selection"},
    "Boost_avg": {"strategy": "weighted_residual_voting"},
    "DT": {},
    "NEQ_Local_selection_5": {"strategy": "selection", "k": 5},
}


class Linear(BaseEstimator):

    def __init__(self, k=5):
        self.k = k
        self.ohe = OneHotEncoder()

    def fit(self, X, y):

        y_ohe = self.ohe.fit_transform(y.reshape(-1, 1)).toarray()
        self.W, _, _, _ = np.linalg.lstsq(X, y_ohe, rcond=None)
        return self

    def predict_proba(self, X):
        # print('X', X.shape)
        # print('w', self.W.shape)

        return X @ self.W

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


set_seeds(random_state)
th.set_float32_matmul_precision("medium")

path_to_data_summary = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
dataset_df = pd.read_csv(cached_path.cached_path(path_to_data_summary), sep="\t")

classification_datasets = dataset_df[
    # (dataset_df["n_binary_features"] == dataset_df["n_features"])
    (dataset_df["task"] == "classification")
    & (dataset_df["n_classes"] == 2)
    # & (dataset_df["n_features"] <= 100)
    & (dataset_df["n_instances"] >= 130)
    & (dataset_df["n_instances"] <= 10000)
]

classification_datasets = classification_datasets.sort_values("n_instances")["dataset"]

print(len(classification_datasets))

res = []
already_trained = False
already_trained_linear = False
for dataset_index, classification_dataset in enumerate(classification_datasets[:]):

    print(
        f"{classification_dataset} ({dataset_index + 1}/{len(classification_datasets) + 1})"
    )
    X, y = fetch_data(classification_dataset, return_X_y=True)
    y_ohe = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    n_neigh = 100  # int(0.1*X.shape[0])
    A = kneighbors_graph(X, n_neighbors=n_neigh, mode="distance", include_self=True)
    # A.data = A.data.max() - A.data + 0.0001
    threshold = np.percentile(A.data, 10)
    A[A > threshold] = 0
    A[A > 0] = 1

    if y.max() != 1 or y.min() != 0:
        for wanted, actual in enumerate(np.unique(y)):
            y[y == actual] = wanted

    edge_homophily = get_edge_homophily(A, y)
    pos_class_ratio = np.bincount(y)[1] / len(y)

    for model_name, model_kwargs in models.items():
        y_pred = []

        print(model_name)
        if "GraphAny" in model_name:
            y_true = []
            for train_indices, test_indices in cv.split(X, y):
                X_train, y_train, y_train_ohe = (
                    X[train_indices],
                    y[train_indices],
                    y_ohe[train_indices],
                )
                X_test, y_test, y_test_ohe = (
                    X[test_indices],
                    y[test_indices],
                    y_ohe[test_indices],
                )

                y_true.append(y_test)

                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    y_train_ohe,
                    y_val_ohe,
                    train_ind_new,
                    val_ind_new,
                ) = train_test_split(
                    X_train,
                    y_train,
                    y_train_ohe,
                    np.arange(y_train.shape[0]),
                    stratify=y_train,
                    random_state=42,
                    test_size=0.1,
                )

                wanted_names = list(["linear", "h_1", "l_1"])

                train_dataset = TensorDataset(
                    th.arange(X_train.shape[0]), th.tensor(y_train_ohe)
                )
                train_dataloader = DataLoader(
                    train_dataset, batch_size=128, num_workers=8
                )

                val_dataset = TensorDataset(
                    th.arange(X_val.shape[0]), th.tensor(y_val_ohe)
                )
                val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8)

                test_dataset = TensorDataset(
                    th.arange(X_test.shape[0]), th.tensor(y_test_ohe)
                )
                test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)

                A_train = A.copy().tolil()
                not_used_in_train = np.concatenate(
                    (test_indices, train_indices[val_ind_new])
                )
                A_train[not_used_in_train, :] = 0
                A_train[:, not_used_in_train] = 0
                A_train = A_train.tocsr()

                A_val = A.copy().tolil()
                not_used_in_val = test_indices
                A_val[not_used_in_val, :] = 0
                A_val[:, not_used_in_val] = 0
                A_val = A_val.tocsr()

                A_test = A.copy()

                # if 'trans' in model_name:
                global logit_dicts_train
                global logit_dicts_val
                global logit_dicts_test

                logit_dicts_train = generate_logit_dicts(
                    X, train_indices[train_ind_new], A_train, y_train_ohe, wanted_names
                )
                logit_dicts_val = generate_logit_dicts(
                    X, train_indices[val_ind_new], A_val, y_val_ohe, wanted_names
                )
                logit_dicts_test = generate_logit_dicts(
                    X, test_indices, A_test, y_test_ohe, wanted_names
                )

                if "ind" in model_name:
                    if already_trained:
                        cur_pred = generate_pred(
                            trainer_ind, model_ind, test_dataloader
                        )
                    else:
                        trainer_ind, model_ind = train_graphany(
                            logit_dicts_train, logit_dicts_val, logit_dicts_test
                        )
                        cur_pred = generate_pred(
                            trainer_ind, model_ind, test_dataloader
                        )
                        already_trained = True
                elif "linear" in model_name:
                    if already_trained_linear:
                        cur_pred = generate_pred(
                            trainer_ind_linear, model_ind_linear, test_dataloader
                        )
                    else:
                        trainer_ind_linear, model_ind_linear = train_graphany(
                            logit_dicts_train, logit_dicts_val, logit_dicts_test
                        )
                        cur_pred = generate_pred(
                            trainer_ind_linear, model_ind_linear, test_dataloader
                        )
                        already_trained_linear = True
                elif "trans" in model_name:
                    trainer, model = train_graphany(
                        logit_dicts_train, logit_dicts_val, logit_dicts_test
                    )
                    total_params_budget = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    cur_pred = generate_pred(trainer, model, test_dataloader)
                elif "mlp" in model_name:
                    trainer_mlp, model_mlp = train_mlp(
                        num_feats=X.shape[1], total_budget=total_params_budget
                    )
                    cur_pred = generate_pred(trainer_mlp, model_mlp, test_dataloader)
                else:
                    raise NotImplementedError(f"{model_name} not understood")

                    # raise NotImplementedError
                # X_train, X_val, y_train, y_val, y_train_ohe, y_val_ohe, train_ind_new, val_ind_new = train_test_split(X_train, y_train, y_train_ohe, np.arange(y_train.shape[0]), stratify=y_train, random_state=42, test_size=0.1)

                y_pred.append(cur_pred)

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            acc = accuracy_score(y_true, y_pred)
            (prec, rec, f1, sup) = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )
        else:
            if "DT" in model_name:
                clf = DecisionTreeClassifier(random_state=random_state)
            if "RF" in model_name:
                clf = RandomForestClassifier(random_state=random_state)
            elif "LR" in model_name:
                clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
            # elif 'NEQ' in model_name:
            #     clf = Pipeline([
            #         ('sc', StandardScaler()),
            #         ('lr', Linear())
            #     ])
            elif "Boost" in model_name:
                clf = Pipeline([("lr", NEQBoost(**model_kwargs))])
            elif "Local" in model_name:
                clf = Pipeline([("lr", NEQ_Local(**model_kwargs))])

            elif "kNN" in model_name:
                clf = KNeighborsClassifier(n_neighbors=n_neigh, weights="distance")
            y_pred = cross_val_predict(clf, X, y, cv=cv)
            acc = accuracy_score(y, y_pred)
            (prec, rec, f1, sup) = precision_recall_fscore_support(
                y, y_pred, average="binary"
            )

        # time_end = time.time() - time_s
        res.append(
            (
                classification_dataset,
                X.shape[0],
                X.shape[1],
                pos_class_ratio,
                edge_homophily,
                model_name,
                acc,
                prec,
                rec,
                f1,
                sup,
            )
        )
        print(res[-1])
        res_df = pd.DataFrame(
            res,
            columns=[
                "dataset",
                "num_samples",
                "num_features",
                "pos_class_ratio",
                "edge_homophily",
                "model",
                "acc",
                "pr",
                "rec",
                "f1",
                "sup",
            ],
        )
        res_df.to_csv(path_to_csv, index=False)

res = pd.DataFrame(
    res,
    columns=[
        "dataset",
        "num_samples",
        "num_features",
        "pos_class_ratio",
        "edge_homophily",
        "model",
        "acc",
        "pr",
        "rec",
        "f1",
        "sup",
    ],
)
res.to_csv(path_to_csv, index=False)

# Step 2: Sort each group by 'f1'
sorted_df = (
    res.groupby("dataset")
    .apply(lambda x: x.sort_values(by="f1", ascending=False))
    .reset_index(drop=True)
)

# Step 3: Assign ranks within each group
sorted_df["rank"] = sorted_df.groupby("dataset").cumcount() + 1

# Step 4: Calculate mean rank for each model across all datasets
mean_ranks = (
    sorted_df.groupby("model")["rank"].mean().reset_index().sort_values(by="rank")
)

print(mean_ranks)
