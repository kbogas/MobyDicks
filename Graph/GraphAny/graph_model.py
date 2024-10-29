import numpy as np
from sklearn.neighbors import kneighbors_graph
from sympy import N
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import lightning as L
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._utils import (
    _binary_search_perplexity as sklearn_binary_search_perplexity,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_entropy_normed_cond_gaussian_prob(X, entropy, metric="euclidean"):
    """
    Parameters
    ----------
    X:              The matrix for pairwise similarity
    entropy:     Perplexity of the conditional prob distribution
    Returns the entropy-normalized conditional gaussian probability based on distances.
    -------
    """

    # Compute pairwise distances
    perplexity = np.exp2(entropy)
    distances = pdist(X, metric=metric)
    distances = squareform(distances)

    # Compute the squared distances
    distances **= 2
    distances = distances.astype(np.float32)
    return sklearn_binary_search_perplexity(distances, perplexity, verbose=0)


def generate_conv(X, A, name='linear'):
    
    F = None
    if name == 'linear':
       F = X 
    elif name.startswith('l_'):
        k_hop = int(name.split('_')[1])
        A_fin = A.copy()
        for k in range(k_hop):
            A_fin = A_fin @ A
        F = A_fin@X
    elif name.startswith('h_'):
        k_hop = int(name.split('_')[1])
        A_fin = np.eye(A.shape[0]) - A
        for k in range(k_hop):
            A_fin = A_fin @ (np.eye(A.shape[0]) - A)
        F = A_fin@X
    else:
        
        raise NotImplementedError(f"{name} not understood")
    return F

def generate_logit_dicts(X, ind_to_focus, A, y_ohe, names):
    logit_dicts = {}
    for name in names:
        logit_dicts[name] = {}
        F = generate_conv(X, A, name)
        F = F[ind_to_focus]
        W, _, _, _  = np.linalg.lstsq(F, y_ohe, rcond=None)
        y_pred = F@W
        # print(name, F.shape, W.shape, y_pred.shape)
        logit_dicts[name]['F'] = F
        logit_dicts[name]['W'] = W
        logit_dicts[name]['y'] = y_pred
    return logit_dicts
        

# def train():
#     pass

# def test():
#     pass        

class GraphAny(nn.Module):
    def __init__(
        self,
        n_hidden,
        feat_channels,
        pred_channels,
        att_temperature,
        entropy=1,
        n_mlp_layer=2,
        device='cuda:0',
        **kwargs
    ):
        super(GraphAny, self).__init__()
        self.feat_channels = feat_channels
        self.pred_channels = pred_channels
        self.entropy = entropy
        self.att_temperature = att_temperature
        self.device = device

        self.dist_feat_dim = len(feat_channels) * (len(feat_channels) - 1)
        self.mlp = MLP(self.dist_feat_dim, n_hidden, len(pred_channels), n_mlp_layer)
    #     self.logit_dicts = {}

    # def calc_logit_dicts(self, X, A, y_ohe):
    #     wanted_names = list(set(self.feat_channels + self.pred_channels))
    #     self.logit_dicts  = generate_logit_dicts(X, A, y_ohe, wanted_names)
    #     return None
            
        
    def compute_dist(self, y_feat):
        bsz, n_channel, n_class = y_feat.shape
        # Conditional gaussian probability
        cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
        for i in range(bsz):
            cond_gaussian_prob[i, :, :] = get_entropy_normed_cond_gaussian_prob(
                y_feat[i, :, :].cpu().numpy(), self.entropy
            )

        # Compute pairwise distances between channels n_channels(n_channels-1)/2 total features
        dist = np.zeros((bsz, self.dist_feat_dim), dtype=np.float32)

        pair_index = 0
        for c in range(n_channel):
            for c_prime in range(n_channel):
                if c != c_prime:  # Diagonal distances are useless
                    dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                    pair_index += 1

        dist = torch.from_numpy(dist).to(y_feat.device)
        return dist

    def forward(self, batch_indices, logit_dicts, dist=None, **kwargs):
        # logit_dict: key: channel, value: prediction of shape (batch_size, n_classes)
        y_feat = torch.stack([torch.tensor(logit_dicts[c]['y'], device=self.device)[batch_indices] for c in self.feat_channels], dim=1)
        y_pred = torch.stack([torch.tensor(logit_dicts[c]['y'], device=self.device)[batch_indices] for c in self.pred_channels], dim=1)

        # ! Fuse y_pred with attentions
        dist = self.compute_dist(y_feat) if dist is None else dist
        # Project pairwise differences to the attention scores (batch_size, n_channels)
        attention = self.mlp(dist)
        attention = th.softmax(attention / self.att_temperature, dim=-1)
        fused_y = th.sum(
            rearrange(attention, "n n_channels -> n n_channels 1") * y_pred, dim=1
        )  # Sum over channels, resulting in (batch_size, n_classes)
        fused_y = th.softmax(fused_y, dim=-1)
        return fused_y, attention.mean(0).tolist()


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        dropout=0.5,
        bias=True,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if n_layers == 1:
            # just linear layer
            self.lins.append(nn.Linear(in_channels, out_channels, bias=bias))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels, bias=bias))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels, bias=bias))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if x.shape[0] > 1:  # Batch norm only if batch_size > 1
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

def get_metrics_from_torch(y_ohe, y_torch):
    y_pred_np = y_torch.argmax(dim=1).cpu().numpy()
    y_true = y_ohe.argmax(dim=1).cpu().numpy()
    pr, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_np, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred_np)
    return acc, pr, rec, f1

from torch import optim

class Lightning_GraphAny(L.LightningModule):
    def __init__(self, n_hidden,feat_channels,pred_channels,att_temperature,entropy=1,n_mlp_layer=2,device='cuda:0',**kwargs):
        super().__init__()
        self.graphany = GraphAny(n_hidden,feat_channels,pred_channels,att_temperature,entropy=1,n_mlp_layer=2,device=device, **kwargs)

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
        loss = nn.functional.mse_loss(y, y_pred)

        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)
        
        values = {"train_loss": loss, "train_f1": f1}  # add more items if needed
        self.log_dict(values, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        
        batch_indices, y = batch
        
        y_pred, att = self.graphany(batch_indices, logit_dicts_val)
        loss = nn.functional.mse_loss(y, y_pred)
        acc, pr, rec, f1 = get_metrics_from_torch(y, y_pred)
        self.log_dict({'val_loss':loss, 'val_f1':f1}, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        
        batch_indices, y = batch
        
        y_pred, att = self.graphany(batch_indices, logit_dicts_test)
        loss = nn.functional.mse_loss(y, y_pred)
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
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=1e-3,
            weight_decay=0.02,
            )
        return optimizer
    

    # def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.02)
        #scheduler = ReduceLROnPlateau(optimizer, 'min')
        
        return {"optimizer": optimizer} 
                #"scheduler": scheduler, 
                # "monitor":"val_loss"}

    
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, train_test_split    
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    import pandas as pd
    from lightning.pytorch import seed_everything
    from sklearn.preprocessing import OneHotEncoder
    import torch as th
    import lightning as L
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks import RichProgressBar
    from pmlb import fetch_data
    
    
    seed_everything(42, workers=True)
    
    #X, y = load_breast_cancer(return_X_y=True)
    X, y = fetch_data('cars', return_X_y=True)
    A =  kneighbors_graph(X, 
                          n_neighbors=100, 
                          mode='distance', 
                          include_self=True)
    #A.data = A.data.max() - A.data + 0.0001
    threshold = np.percentile(A.data, 10)
    A[A>threshold] = 0
    A[A > 0] = 1
    dataset_name = 'breast_cancer'
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    
    splits = []
    for split_id, (train_ind, test_ind) in enumerate(cv.split(X, y)):
        
        ohe = OneHotEncoder()
        y_ohe = ohe.fit_transform(y.reshape(-1,1)).todense()
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        dt = Pipeline([
            ('sc', StandardScaler()),
            ('lr', LogisticRegression(C=0.0001, max_iter=1000))
        ])
        
        #dt = LogisticRegression(max_iter=1000)#DecisionTreeClassifier(random_state=42)
        
        X_train, y_train, y_train_ohe = X[train_ind], y[train_ind], y_ohe[train_ind]
        X_train, X_val, y_train, y_val, y_train_ohe, y_val_ohe, train_ind_new, val_ind_new = train_test_split(X_train, y_train, y_train_ohe, np.arange(y_train.shape[0]), stratify=y_train, random_state=42, test_size=0.1)
        #A_val = A_train[val_ind_new, :][:, val_ind_new]
        #A_train = A_train[train_ind_new, :][:, train_ind_new]
        
        X_test, y_test, y_test_ohe = X[test_ind], y[test_ind], y_ohe[test_ind]
        
        
        train_dataset = TensorDataset( th.arange(X_train.shape[0]), th.tensor(y_train_ohe))
        train_dataloader = DataLoader(train_dataset, batch_size= 128)

        val_dataset = TensorDataset( th.arange(X_val.shape[0]), th.tensor(y_val_ohe))
        val_dataloader = DataLoader(val_dataset, batch_size= 32)
        
        test_dataset = TensorDataset( th.arange(X_test.shape[0]), th.tensor(y_test_ohe))
        test_dataloader = DataLoader(test_dataset, batch_size= 32)
        
        
        device = 'cuda:0'
        model = Lightning_GraphAny(n_hidden=128, 
                           feat_channels=[
                               'linear',
                                'h_1', 
                                #'h_2',
                                'l_1',
                                #'l_2'
                                ], 
                           pred_channels=[
                               'linear',
                                'h_1', 
                                #'h_2',
                                'l_1',
                                #'l_2'
                                ],
                           att_temperature=5, 
                           entropy=1,
                           mlp_layer=2,
                           device=device
                           )
        
        
        wanted_names = list(set(model.graphany.feat_channels + model.graphany.pred_channels))
        # print(y_train_ohe.shape)
        
        # for name, param in model.named_parameters():
        #     print(name)
        #     print(param.shape)
        #     print()
        # exit()
        
        # Mask adjacencies for train,val,test
        A_train = A.copy()
        not_used_in_train = np.concatenate((test_ind, train_ind[val_ind_new]))
        A_train[not_used_in_train, :] = 0
        A_train[:, not_used_in_train] = 0
        
        A_val = A.copy()
        not_used_in_val = test_ind
        A_val[not_used_in_val, :] = 0
        A_val[:, not_used_in_val] = 0
        
        A_test = A.copy()
        
        logit_dicts_train = generate_logit_dicts(X, train_ind[train_ind_new], A_train, y_train_ohe, wanted_names)
        logit_dicts_val = generate_logit_dicts(X, train_ind[val_ind_new], A_val, y_val_ohe, wanted_names)
        logit_dicts_test = generate_logit_dicts(X, test_ind, A_test, y_test_ohe, wanted_names)
    
       
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        
        trainer = L.Trainer(
                            
                            deterministic=True, 
                            devices=[0], 
                            accelerator='gpu', 
                            max_epochs=500, 
                            #enable_model_summary=True, 
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001), RichProgressBar(leave=True)], 
                            check_val_every_n_epoch=5)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        predictions = trainer.predict(model=model, dataloaders=test_dataloader)
        predictions = th.vstack(predictions)
        y_pred = predictions.argmax(dim=1).detach().numpy()
        pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        print(f"GraphAny F1: {f1:.4f}")

        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)
        X = sc.transform(X)
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print(f"SC+LR ULTRA DIFFICULT F1: {f1:.4f}")
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        pr, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print(f"DT ULTRA DIFFICULT F1: {f1:.4f}")
        
        
        splits.append((dataset_name, split_id, acc, pr, rec, f1))
    df = pd.DataFrame(splits, columns=['dataset', 'split_id', 'acc', 'pr', 'rec', 'f1'])
    print(df)
    print(df[df.columns[2:]].mean(axis=0))
        # optimizer = model.configure_optimizers()
        # for batch_idx, batch in enumerate(train_dataloader):
        #     batch_indices, y_train_batch, input_type = batch
        #     loss = model.training_step(batch)

        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
            

