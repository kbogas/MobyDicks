import pandas as pd
import cached_path
from pmlb import fetch_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_predict
import time
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from scipy.special import softmax
from sklearn.base import clone
from tree_embedding_nn import TreeNetClassifier
import pytorch_lightning as pl


random_state = 42
pl.seed_everything(random_state)


path_to_data_summary = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
dataset_df = pd.read_csv(cached_path.cached_path(path_to_data_summary), sep="\t")

classification_datasets = dataset_df[
    # (dataset_df["n_binary_features"] == dataset_df["n_features"])
    (dataset_df["task"] == "classification")
    & (dataset_df["n_classes"] == 2)
    # & (dataset_df["n_features"] <= 150)
    & (dataset_df["n_features"] >= 10)
    & (dataset_df["n_instances"] > 100)
]["dataset"][:]

print(len(classification_datasets))

models = {
    "Baseline": {},
    "Tree_NN": {"concat": False},
    "Tree_NN_Concat": {"concat": True},
}


number_of_cv_folds = 5
num_estimators = 100
max_depth = None

cv = StratifiedKFold(number_of_cv_folds, random_state=random_state, shuffle=True)
base_class = RandomForestClassifier(
    n_estimators=num_estimators, max_depth=max_depth, random_state=42
)
##DecisionTreeClassifier(max_depth=None, random_state=42)#

res = []
for dataset_index, classification_dataset in enumerate(
    classification_datasets[::-1][:]
):

    print(
        f"{classification_dataset} ({dataset_index + 1}/{len(classification_datasets) + 1})"
    )
    if "deprecated" in classification_dataset:
        print(f"Skipping {classification_dataset} as deprecated from PMLB...")
        continue
    try:
        X, y = fetch_data(classification_dataset, return_X_y=True)
    except ValueError as e:
        print(
            f"Probably not found dataset {classification_dataset} in PMLB and skipping...\n {e}"
        )
        continue
    if y.max() != 1 or y.min() != 0:
        for wanted, actual in enumerate(np.unique(y)):
            y[y == actual] = wanted

    imb_ratio = np.bincount(y).max() / np.bincount(y).min()
    print(f"{X.shape} with ratio : {imb_ratio:.4f}\n")

    for model_name, model_kwargs in models.items():
        y_pred = np.empty_like(y)
        sample_weights = None
        time_s = time.time()
        for train_indices, test_indices in cv.split(X, y):
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            X_train_filtered = X_train.copy()
            y_train_filtered = y_train.copy()
            if model_name.startswith("Tree_NN"):
                clf = TreeNetClassifier(
                    n_estimators=num_estimators,
                    max_depth=max_depth,
                    mlp_hidden_dims=[32, 16],
                    lr=0.005,
                    epochs=100,
                    patience=3,
                    batch_size=256,
                    check_val_every_n_epoch=5,
                    concat_original_features=model_kwargs[
                        "concat"
                    ],  # Try with False to see the difference
                    device="auto",
                )
            else:
                clf = clone(base_class)
            # print(model_name, X_train_filtered.shape[0])
            clf.fit(X_train_filtered, y_train_filtered)
            y_pred_cur = clf.predict(X_test)

            y_pred[test_indices] = y_pred_cur
            # print(f'TRUE', y_test)

        acc = accuracy_score(y, y_pred)
        (prec, rec, f1, sup) = precision_recall_fscore_support(
            y, y_pred, average="binary"
        )

        print(model_name)
        print(classification_report(y, y_pred))
        time_end = time.time() - time_s

        res.append(
            (
                classification_dataset,
                imb_ratio,
                model_name,
                time_end,
                acc,
                prec,
                rec,
                f1,
            )
        )

res = pd.DataFrame(
    res,
    columns=["dataset", "dataset_class_imb", "model", "time", "acc", "pr", "rec", "f1"],
)

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
res.to_csv("./results/tree_emb_res_depth_None.csv", index=False)
