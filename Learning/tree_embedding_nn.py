# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
# from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris, load_breast_cancer
# from sklearn.metrics import accuracy_score

# # For reproducibility
# pl.seed_everything(42)
# torch.set_float32_matmul_precision("medium")


# # --- PyTorch Lightning Module for the MLP (Modified) ---
# class MLPModule(pl.LightningModule):
#     # ADDED `check_val_every_n_epoch` to the constructor
#     def __init__(
#         self,
#         input_dim,
#         hidden_dims,
#         output_dim,
#         lr,
#         weight_decay,
#         dropout_p,
#         check_val_every_n_epoch=1,
#     ):
#         super().__init__()
#         # `save_hyperparameters` makes all arguments available via self.hparams
#         self.save_hyperparameters()

#         layers = []
#         last_dim = input_dim
#         for h_dim in hidden_dims:
#             layers.append(nn.Linear(last_dim, h_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(h_dim))
#             layers.append(nn.Dropout(dropout_p))
#             last_dim = h_dim

#         layers.append(nn.Linear(last_dim, output_dim))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

#     def _common_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         return loss, logits

#     def training_step(self, batch, batch_idx):
#         loss, _ = self._common_step(batch, batch_idx)
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, _ = self._common_step(batch, batch_idx)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def predict_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         return F.softmax(logits, dim=1)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.lr,
#             weight_decay=self.hparams.weight_decay,
#         )
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             "min",
#             factor=0.1,
#             patience=5,
#         )

#         # THIS IS THE FIX: Align the scheduler's frequency with the validation frequency
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",
#                 "interval": "epoch",
#                 "frequency": self.hparams.check_val_every_n_epoch,
#             },
#         }


# # --- Main Scikit-Learn Compatible Classifier (Modified) ---
# class TreeNetClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(
#         self,
#         # RF Hyperparameters
#         n_estimators=100,
#         max_depth=5,
#         # MLP Hyperparameters
#         mlp_hidden_dims=[64, 32],
#         mlp_dropout=0.2,
#         # Training Hyperparameters
#         lr=1e-3,
#         weight_decay=1e-4,
#         batch_size=32,
#         epochs=100,
#         check_val_every_n_epoch=1,
#         # Config
#         concat_original_features=True,
#         val_split_ratio=0.1,
#         patience=10,
#         device="auto",
#         random_state=42,
#     ):

#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.mlp_hidden_dims = mlp_hidden_dims
#         self.mlp_dropout = mlp_dropout
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.check_val_every_n_epoch = check_val_every_n_epoch
#         self.concat_original_features = concat_original_features
#         self.val_split_ratio = val_split_ratio
#         self.patience = patience
#         self.device = device
#         self.random_state = random_state

#     def _get_device(self):
#         if self.device == "auto":
#             return "gpu" if torch.cuda.is_available() else "cpu"
#         return self.device

#     def fit(self, X, y):
#         self.classes_, y_encoded = np.unique(y, return_inverse=True)
#         self.n_classes_ = len(self.classes_)

#         if self.n_classes_ < 2:
#             raise ValueError("This classifier requires at least 2 classes.")

#         X_np = X.values if hasattr(X, "values") else X
#         y_np = y_encoded

#         # 1. Train the Random Forest
#         print("--- Fitting Random Forest ---")
#         self.rf_ = RandomForestClassifier(
#             n_estimators=self.n_estimators,
#             max_depth=self.max_depth,
#             random_state=self.random_state,
#         )
#         self.rf_.fit(X_np, y_np)

#         # 2. Create the leaf probability mapping
#         print("--- Creating Leaf Probability Embeddings ---")
#         self.leaf_prob_maps_ = []
#         for estimator in self.rf_.estimators_:
#             tree = estimator.tree_
#             leaf_values = tree.value
#             leaf_probs = (leaf_values[:, 0, 1] + 1e-6) / (
#                 leaf_values.sum(axis=2).flatten() + 2e-6
#             )
#             node_to_prob = {i: prob for i, prob in enumerate(leaf_probs)}
#             self.leaf_prob_maps_.append(node_to_prob)

#         # 3. Transform the data
#         print("--- Transforming Data with Forest ---")
#         X_transformed = self._transform_features(X_np)

#         # 4. Scale original features if concatenating
#         if self.concat_original_features:
#             self.scaler_ = StandardScaler()
#             scaled_original = self.scaler_.fit_transform(X_np)
#             X_final = np.hstack([scaled_original, X_transformed])
#         else:
#             X_final = X_transformed

#         self.input_dim_ = X_final.shape[1]

#         # 5. Train the MLP using PyTorch Lightning
#         print(f"--- Fitting MLP (Input Dim: {self.input_dim_}) ---")

#         X_train, X_val, y_train, y_val = train_test_split(
#             X_final,
#             y_np,
#             test_size=self.val_split_ratio,
#             random_state=self.random_state,
#             stratify=y_np,
#         )

#         train_dataset = TensorDataset(
#             torch.FloatTensor(X_train), torch.LongTensor(y_train)
#         )
#         val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

#         train_loader = DataLoader(
#             train_dataset, batch_size=self.batch_size, shuffle=True
#         )
#         val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

#         # THIS IS THE FIX: Pass the validation frequency to the MLP module
#         self.mlp_ = MLPModule(
#             input_dim=self.input_dim_,
#             hidden_dims=self.mlp_hidden_dims,
#             output_dim=self.n_classes_,
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#             dropout_p=self.mlp_dropout,
#             check_val_every_n_epoch=self.check_val_every_n_epoch,
#         )

#         early_stop_callback = EarlyStopping(
#             monitor="val_loss", patience=self.patience, verbose=False, mode="min"
#         )

#         progress_bar = RichProgressBar(
#             theme=RichProgressBarTheme(
#                 description="green_yellow",
#                 progress_bar="green1",
#                 progress_bar_finished="green1",
#             )
#         )

#         trainer = pl.Trainer(
#             max_epochs=self.epochs,
#             accelerator=self._get_device(),
#             callbacks=[early_stop_callback, progress_bar],
#             check_val_every_n_epoch=self.check_val_every_n_epoch,
#             enable_checkpointing=False,
#             logger=False,
#             enable_model_summary=False,
#         )

#         trainer.fit(self.mlp_, train_loader, val_loader)

#         self.mlp_.eval()
#         print("--- Fit Complete ---")
#         return self

#     def _transform_features(self, X):
#         X_np = X.values if hasattr(X, "values") else X
#         leaf_indices_per_tree = self.rf_.apply(X_np)

#         prob_features = np.zeros_like(leaf_indices_per_tree, dtype=float)
#         for i, tree_map in enumerate(self.leaf_prob_maps_):
#             prob_features[:, i] = [
#                 tree_map.get(leaf_idx, 0.5) for leaf_idx in leaf_indices_per_tree[:, i]
#             ]

#         return prob_features

#     def predict_proba(self, X):
#         X_np = X.values if hasattr(X, "values") else X
#         X_transformed = self._transform_features(X_np)

#         if self.concat_original_features:
#             scaled_original = self.scaler_.transform(X_np)
#             X_final = np.hstack([scaled_original, X_transformed])
#         else:
#             X_final = X_transformed

#         X_tensor = torch.FloatTensor(X_final).to(self.mlp_.device)

#         with torch.no_grad():
#             self.mlp_.eval()
#             probas = F.softmax(self.mlp_(X_tensor), dim=1)

#         return probas.cpu().numpy()

#     def predict(self, X):
#         probas = self.predict_proba(X)
#         return self.classes_[np.argmax(probas, axis=1)]


# if __name__ == "__main__":
#     # --- Example Usage on Iris Dataset ---
#     print("🚀 Running TreeNetClassifier on Breast Cancer Dataset 🚀")

#     # 1. Load and prepare data for binary classification
#     # We'll classify: class 0 (Setosa) vs. class 1+2 (Not Setosa)
#     X, y = load_breast_cancer(return_X_y=True)
#     y_binary = (y > 0).astype(int)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
#     )

#     # 2. Initialize and fit the classifier
#     # Using a shallow depth to keep the example fast
#     treenet = TreeNetClassifier(
#         n_estimators=100,
#         max_depth=10,
#         mlp_hidden_dims=[32, 16],
#         lr=0.005,
#         epochs=100,
#         patience=7,
#         concat_original_features=False,  # Try with False to see the difference
#         device="auto",
#     )
#     treenet.fit(X_train, y_train)

#     # 3. Make predictions and evaluate
#     y_pred = treenet.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     print("\n✅ Test Complete!")
#     print(f"TreeNetClassifier Accuracy: {accuracy:.4f}")

#     # For comparison, let's test a standard RandomForest
#     rf_comp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
#     rf_comp.fit(X_train, y_train)
#     y_pred_rf = rf_comp.predict(X_test)
#     accuracy_rf = accuracy_score(y_test, y_pred_rf)
#     print(f"Standard RandomForest Accuracy: {accuracy_rf:.4f}")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score


torch.set_float32_matmul_precision("medium")


# --- PyTorch Lightning Module for the MLP (Refactored for Embeddings) ---
class MLPModule(pl.LightningModule):
    def __init__(
        self,
        n_numerical_features,
        embedding_specs,  # List of (n_categories, embedding_dim)
        hidden_dims,
        output_dim,
        lr,
        weight_decay,
        dropout_p,
        check_val_every_n_epoch=1,
        class_weights=[1, 1],
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create embedding layers for each categorical feature (each tree)
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(n_cat, n_dim) for n_cat, n_dim in embedding_specs]
        )

        # Calculate the total size of the concatenated embedding vectors
        total_embedding_dim = sum(n_dim for _, n_dim in embedding_specs)

        # Batch norm for the numerical features
        self.numerical_batch_norm = nn.BatchNorm1d(n_numerical_features)

        # Define the input dimension for the main MLP tower
        combined_input_dim = total_embedding_dim + n_numerical_features

        # Create the main MLP tower
        layers = []
        last_dim = combined_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_p))
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.mlp_tower = nn.Sequential(*layers)

    def forward(self, x_categorical, x_numerical):
        # Process categorical features through their embedding layers
        embedded_vectors = []
        for i, emb_layer in enumerate(self.embedding_layers):
            # x_categorical is (batch, n_trees), select column i for the i-th tree
            embedded_vectors.append(emb_layer(x_categorical[:, i]))

        concatenated_embeddings = torch.cat(embedded_vectors, dim=1)

        # Process numerical features
        normed_numerical = self.numerical_batch_norm(x_numerical)

        # Combine embedded categorical and numerical features
        combined_features = torch.cat(
            [concatenated_embeddings, normed_numerical], dim=1
        )

        # Pass through the final MLP tower
        return self.mlp_tower(combined_features)

    def _common_step(self, batch, batch_idx):
        # The batch now contains three elements
        x_categorical, x_numerical, y = batch
        logits = self(x_categorical, x_numerical)
        loss = F.cross_entropy(
            logits, y, weight=torch.Tensor(self.hparams.class_weights).to(logits.device)
        )
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x_categorical, x_numerical, y = batch
        logits = self(x_categorical, x_numerical)
        return F.softmax(logits, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": self.hparams.check_val_every_n_epoch,
            },
        }


# --- Main Scikit-Learn Compatible Classifier (Refactored for Embeddings) ---
class TreeNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        # RF Hyperparameters
        n_estimators=100,
        max_depth=10,
        # Embedding Hyperparameters
        embedding_dim=4,
        # MLP Hyperparameters
        mlp_hidden_dims=[64, 32],
        mlp_dropout=0.2,
        # Training Hyperparameters
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        epochs=100,
        check_val_every_n_epoch=1,
        class_weights=[1, 1],
        # Config
        concat_original_features=True,
        val_split_ratio=0.1,
        patience=10,
        device="auto",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weights = class_weights
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.concat_original_features = concat_original_features
        self.val_split_ratio = val_split_ratio
        self.patience = patience
        self.device = device
        self.random_state = random_state

    def _get_device(self):
        if self.device == "auto":
            return "gpu" if torch.cuda.is_available() else "cpu"
        return self.device

    def fit(self, X, y):
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError("This classifier requires at least 2 classes.")

        X_np = X.values if hasattr(X, "values") else X
        y_np = y_encoded

        # 1. Train the Random Forest
        print("--- Fitting Random Forest ---")
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.rf_.fit(X_np, y_np)

        # 2. Create mappings and get embedding specifications
        print("--- Creating Leaf Mappings & Embedding Specs ---")
        self.leaf_prob_maps_ = []
        self.embedding_specs_ = []
        for estimator in self.rf_.estimators_:
            tree = estimator.tree_
            # Get number of leaves for this tree to define vocab size
            n_leaves = tree.n_leaves
            self.embedding_specs_.append((n_leaves, self.embedding_dim))

            # Create the probability map
            leaf_values = tree.value
            leaf_probs = (leaf_values[:, 0, 1] + 1e-6) / (
                leaf_values.sum(axis=2).flatten() + 2e-6
            )
            node_to_prob = {i: prob for i, prob in enumerate(leaf_probs)}
            self.leaf_prob_maps_.append(node_to_prob)

        # 3. Transform data into its numerical and categorical parts
        print("--- Transforming Data with Forest ---")
        prob_features, index_features = self._transform_features(X_np)

        numerical_features = [prob_features]
        if self.concat_original_features:
            self.scaler_ = Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("imp", SimpleImputer(strategy="constant", fill_value=0)),
                ]
            )
            scaled_original = self.scaler_.fit_transform(X_np)
            numerical_features.insert(0, scaled_original)

        X_numerical_final = np.hstack(numerical_features)
        X_categorical_final = index_features

        # 5. Train the MLP using PyTorch Lightning
        print(f"--- Fitting MLP ---")

        # Split data for early stopping
        indices = np.arange(X_np.shape[0])
        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_split_ratio,
            random_state=self.random_state,
            stratify=y_np,
        )

        train_dataset = TensorDataset(
            torch.LongTensor(X_categorical_final[train_indices]),
            torch.FloatTensor(X_numerical_final[train_indices]),
            torch.LongTensor(y_np[train_indices]),
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_categorical_final[val_indices]),
            torch.FloatTensor(X_numerical_final[val_indices]),
            torch.LongTensor(y_np[val_indices]),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.mlp_ = MLPModule(
            n_numerical_features=X_numerical_final.shape[1],
            embedding_specs=self.embedding_specs_,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=self.n_classes_,
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout_p=self.mlp_dropout,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            class_weights=self.class_weights,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=self.patience, verbose=False, mode="min"
        )
        progress_bar = RichProgressBar()

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=self._get_device(),
            callbacks=[early_stop_callback, progress_bar],
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
        )

        trainer.fit(self.mlp_, train_loader, val_loader)
        self.mlp_.eval()
        print("--- Fit Complete ---")
        return self

    def _transform_features(self, X):
        """Returns both probability and index features."""
        X_np = X.values if hasattr(X, "values") else X
        # This gives the node index each sample falls into for each tree
        # For RF, node indices are not contiguous for leaves, so we need to map them
        node_indices_per_tree = self.rf_.apply(X_np)

        prob_features = np.zeros_like(node_indices_per_tree, dtype=float)
        # We need a consistent leaf index from 0 to n_leaves-1 for the embedding layer
        leaf_indices_final = np.zeros_like(node_indices_per_tree, dtype=int)

        for i, estimator in enumerate(self.rf_.estimators_):
            tree_map_prob = self.leaf_prob_maps_[i]
            # Create a mapping from arbitrary node index to a clean 0-based leaf index
            leaf_nodes = np.where(estimator.tree_.feature == -2)[0]
            node_to_leaf_idx_map = {
                node_id: leaf_id for leaf_id, node_id in enumerate(leaf_nodes)
            }

            # Apply the mappings
            prob_features[:, i] = [
                tree_map_prob.get(node_idx, 0.5)
                for node_idx in node_indices_per_tree[:, i]
            ]
            leaf_indices_final[:, i] = [
                node_to_leaf_idx_map.get(node_idx, 0)
                for node_idx in node_indices_per_tree[:, i]
            ]

        return prob_features, leaf_indices_final

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else X
        prob_features, index_features = self._transform_features(X_np)

        numerical_features = [prob_features]
        if self.concat_original_features:
            scaled_original = self.scaler_.transform(X_np)
            numerical_features.insert(0, scaled_original)

        X_numerical_final = np.hstack(numerical_features)
        X_categorical_final = index_features

        # Convert to tensors
        x_cat_tensor = torch.LongTensor(X_categorical_final).to(self.mlp_.device)
        x_num_tensor = torch.FloatTensor(X_numerical_final).to(self.mlp_.device)

        with torch.no_grad():
            self.mlp_.eval()
            logits = self.mlp_(x_cat_tensor, x_num_tensor)
            probas = F.softmax(logits, dim=1)

        return probas.cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


if __name__ == "__main__":
    # --- Example Usage on Breast Cancer Dataset ---
    print("🚀 Running TreeNetClassifier with Learnable Embeddings 🚀")

    max_depth = None
    n_estimators = 100
    random_state = 42

    # For reproducibility
    pl.seed_everything(random_state)

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    treenet = TreeNetClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,  # Shallower depth -> fewer leaves -> smaller embedding tables
        embedding_dim=100,  # Dimension for the learned leaf vectors
        mlp_hidden_dims=[64, 32],
        lr=0.001,
        epochs=100,
        patience=10,
        concat_original_features=True,
        device="auto",
    )
    treenet.fit(X_train, y_train)

    y_pred = treenet.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Test Complete!")
    print(f"TreeNetClassifier w/ Embeddings Accuracy: {accuracy:.4f}")

    # For comparison, let's test a standard RandomForest
    rf_comp = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    rf_comp.fit(X_train, y_train)
    y_pred_rf = rf_comp.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Standard RandomForest Accuracy: {accuracy_rf:.4f}")
