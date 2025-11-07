from prime_adj.pam_creation import create_pam_matrices
from prime_adj.utils import get_sparsity
from prime_adj.data_loading import load_data
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dgl
import torch

import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score


class TransE(nn.Module):
    def __init__(
        self, num_entities, num_relations, embedding_dim, margin=1.0, p_norm=1
    ):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.p_norm = p_norm
        self.num_entities = num_entities
        self.num_relations = num_relations

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return h_emb, r_emb, t_emb

    def compute_loss(self, h_emb, r_emb, t_emb, neg_h_emb, neg_t_emb):
        pos_score = self.score(h_emb, r_emb, t_emb)
        # print(neg_h_emb.shape, r_emb.shape, t_emb.shape)
        # print(r_emb.reshape(r_emb.shape[0], 1, r_emb.shape[1]).repeat(1,10,1).shape, t_emb.reshape(t_emb.shape[0], 1, t_emb.shape[1]).repeat(1,10,1).shape)
        expanded_r_emb = r_emb.reshape(r_emb.shape[0], 1, r_emb.shape[1]).repeat(
            1, 10, 1
        )
        expanded_h_emb = h_emb.reshape(h_emb.shape[0], 1, h_emb.shape[1]).repeat(
            1, 10, 1
        )
        expanded_t_emb = t_emb.reshape(t_emb.shape[0], 1, t_emb.shape[1]).repeat(
            1, 10, 1
        )
        neg_score_h = self.score(neg_h_emb, expanded_r_emb, expanded_t_emb)
        neg_score_t = self.score(expanded_h_emb, expanded_r_emb, neg_t_emb)

        loss_h = torch.relu(self.margin + pos_score - neg_score_h.mean(1)).mean()
        loss_t = torch.relu(self.margin + pos_score - neg_score_t.mean(1)).mean()
        return loss_h + loss_t

    def get_embeddings(self, h, r, t):
        h_emb = self.entity_embeddings(h) if h is not None else None
        r_emb = self.relation_embeddings(r) if r is not None else None
        t_emb = self.entity_embeddings(t) if t is not None else None
        return h_emb, r_emb, t_emb

    def score(self, h_emb, r_emb, t_emb):
        return torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)


# def extract_neighborhood(graph, node, k=2):
#     """Extracts the k-hop in-neighborhood for a node in a homogeneous graph."""
#     visited = {node.item()}
#     queue = [(node.item(), 0)]  # (node, distance)
#     nodes_to_include = [node.item()]

#     while queue:
#         current_node, distance = queue.pop(0)
#         if distance >= k:
#             continue

#         neighbors = graph.in_edges(current_node)[
#             0
#         ].tolist()  # get source nodes of in_edges
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append((neighbor, distance + 1))
#                 nodes_to_include.append(neighbor)

#     subgraph = dgl.node_subgraph(
#         graph, nodes_to_include, relabel_nodes=False, store_ids=True
#     )
#     return subgraph


def extract_neighborhood(graph, node, k=2):
    # bfs = dgl.sampling.bfs_nodes_generator(graph, node, k)
    # neighborhood_nodes = torch.cat([nodes for nodes in bfs])
    # subgraph = dgl.node_subgraph(graph, neighborhood_nodes)
    # print(node.item())
    subgraph, _ = dgl.khop_in_subgraph(
        graph, nodes=node.item(), k=k, relabel_nodes=False, store_ids=True
    )
    return subgraph


def extract_neighborhood_df(df_train, node, rel, top_k=10):
    # bfs = dgl.sampling.bfs_nodes_generator(graph, node, k)
    # neighborhood_nodes = torch.cat([nodes for nodes in bfs])
    # subgraph = dgl.node_subgraph(graph, neighborhood_nodes)
    subset = df_train[df_train["rel_mapped"] == rel.item()]
    # print(subset)
    subset_only_specific_head = subset[subset["head_mapped"] == node.item()]
    if len(subset_only_specific_head) > 0:
        wanted_triples = subset_only_specific_head[
            ["head_mapped", "rel_mapped", "tail_mapped"]
        ].values.tolist()
    else:
        # wanted_triples = []
        possible_heads = subset["head_mapped"].unique()
        kept_heads = possible_heads[
            torch.argsort(nodesimilarity[node, possible_heads], descending=True)[
                :top_k
            ].numpy()
        ]
        subset = subset[subset["head_mapped"].isin(kept_heads)]
        wanted_triples = subset[
            ["head_mapped", "rel_mapped", "tail_mapped"]
        ].values.tolist()
    return wanted_triples


def generate_negative_samples(graph, batch_size):
    num_neg = NUM_NEGATIVE  # number of negative samples per positive sample
    neg_head = torch.randint(0, graph.num_nodes(), (batch_size, num_neg))
    neg_tail = torch.randint(0, graph.num_nodes(), (batch_size, num_neg))
    return neg_head, neg_tail


from functools import lru_cache


@lru_cache
def adapt_transE(model, query, neighborhood, learning_rate=0.0001, num_steps=10):
    h_test, r_test, t_test = query
    h_test_emb, r_test_emb, t_test_emb = model.get_embeddings(h_test, r_test, t_test)
    adapted_r = r_test_emb.clone().detach().requires_grad_(True)
    if t_test is None:
        adapted_h = h_test_emb.clone().detach().requires_grad_(True)
        adapted_t = model.entity_embeddings.weight.clone().detach()
        optimizer2 = optim.Adam([adapted_h, adapted_r], lr=learning_rate)
    elif h_test is None:
        adapted_h = model.entity_embeddings.weight.clone().detach()
        adapted_t = t_test_emb.clone().detach().requires_grad_(True)
        optimizer2 = optim.Adam([adapted_t, adapted_r], lr=learning_rate)

    for _ in range(num_steps):
        optimizer2.zero_grad()
        loss2 = torch.tensor(0.0)  # , requires_grad=True)
        got = False
        for h_i, r_i, t_i in neighborhood:
            # print("neighborhood")
            h_i_emb, r_i_emb, t_i_emb = model.get_embeddings(
                torch.Tensor([h_i]).int(),
                torch.Tensor([r_i]).int(),
                torch.Tensor([t_i]).int(),
            )

            if t_test is None:
                # loss2 += torch.relu(
                #     model.margin
                #     + torch.norm(adapted_h + r_i_emb - t_i_emb, p=model.p_norm)
                #     - torch.norm(h_i_emb + r_i_emb - t_i_emb, p=model.p_norm)
                # )
                got = True
                if h_i == h_test:
                    loss2 += model.score(
                        adapted_h,
                        adapted_r,
                        t_i_emb,
                    )[0]
                else:
                    loss2 += model.score(h_i_emb, adapted_r, t_i_emb)[0]
                # print("added loss for head", loss2)
            elif h_test is None and t_i == t_test:
                loss2 += torch.norm(
                    h_i_emb + adapted_r - adapted_t, p=model.p_norm, dim=-1
                )
        if got:
            # print(f"Loss2 for {h_test.item()}: {loss2.item()}")
            loss2.backward()
            optimizer2.step()
        # print(f"DIIF: {(adapted_h - h_test_emb).abs().sum()}")

    if t_test is None:
        scores = model.score(adapted_h, adapted_r, model.entity_embeddings.weight)
    elif h_test is None:
        scores = model.score(
            model.entity_embeddings.weight,
            adapted_r,
            adapted_t,
        )
    return scores


def get_scores(ranks):
    mr = torch.mean(ranks).item()
    mrr = torch.mean(1 / ranks).item()
    scores = {"Mean Rank": mr, "MRR": mrr}
    for k in [1, 3, 10, 100]:
        score = (ranks <= k).sum() / len(ranks)
        scores[f"Hits@{k}"] = score.item()
    return scores


def evaluate(model, test_graph, mode="simple"):
    model.eval()
    auc_scores = []

    if mode == "simple":
        with torch.no_grad():
            h, t = test_graph.edges()
            r = test_graph.edata["rel"]
            h_emb, r_emb, t_emb = model(h, r, t)
            number_wanted = model.num_entities

            # print(neg_h_emb.shape, r_emb.shape, t_emb.shape)
            # print(r_emb.reshape(r_emb.shape[0], 1, r_emb.shape[1]).repeat(1,10,1).shape, t_emb.reshape(t_emb.shape[0], 1, t_emb.shape[1]).repeat(1,10,1).shape)
            expanded_r_emb = r_emb.reshape(r_emb.shape[0], 1, r_emb.shape[1]).repeat(
                1, number_wanted, 1
            )
            expanded_h_emb = h_emb.reshape(h_emb.shape[0], 1, h_emb.shape[1]).repeat(
                1, number_wanted, 1
            )
            expanded_t_emb = (
                model.entity_embeddings(torch.arange(number_wanted))
                .reshape(1, number_wanted, h_emb.shape[1])
                .repeat(h_emb.shape[0], 1, 1)
            )
            score_all = model.score(expanded_h_emb, expanded_r_emb, expanded_t_emb)
            argsorted = torch.argsort(score_all, dim=1)
            ranks = (argsorted == t.unsqueeze(-1)).nonzero()[:, 1] + 1
            ranks = ranks.float()
            # print(ranks)

    elif mode == "adapt":
        ranks = []
        for test_index, (h, t, r) in tqdm.tqdm(
            enumerate(
                zip(
                    test_graph.edges()[0],
                    test_graph.edges()[1],
                    test_graph.edata["rel"],
                )
            ),
            total=len(test_graph.edges()[0]),
        ):
            neighborhood = extract_neighborhood_df(df_train, h, r)
            scores = adapt_transE(
                model, (h, r, None), tuple([tuple(l) for l in neighborhood])
            )
            rank = torch.argsort(scores).numpy().tolist().index(t) + 1
            # print(f"OLD:{ranks_simple[test_index]} NEW:{rank}")
            ranks.append(rank)
        ranks = torch.Tensor(ranks).float()
        # print(ranks)

    else:
        raise NotImplementedError()
    scores = get_scores(ranks)
    return scores, ranks


import tqdm

project_to_path = {
    "codex-s": "/home/kbougatiotis/GIT/Prime_Adj/data/codex-s/",
    "codex-l": "/home/kbougatiotis/GIT/Prime_Adj/data/codex-l/",
    "WN18RR": "/home/kbougatiotis/GIT/Prime_Adj/data/WN18RR/",
    "FB15k-237": "../data/FB15k-237/",
    "YAGO3-10-DR": "../data/YAGO3-10-DR/",
    "YAGO3-10": "../data/YAGO3-10",
    "NELL-995": "../data/NELL-995",
    "Simpathic": "/home/kbougatiotis/GIT/PAM_Biomedical/Simpathic/data/simpathic/stratified_folds_big_with_train/split_0/",
    # "hetionet": "./data/Hetionet/hetionet-v1.0-edges.tsv",
    # "ogbl-wikikg2": "path",
}
project_name = "codex-s"
add_inverse_edges = "NO"

df_train_orig, df_train, df_eval, df_test, already_seen_triples_ = load_data(
    project_to_path[project_name], project_name, add_inverse_edges=add_inverse_edges
)


df_all = pd.concat([df_train, df_eval, df_test])
unq_nodes = np.concatenate((df_all["head"].unique(), df_all["tail"].unique()))
unq_rels = df_all["rel"].unique()
node2id = {v: index for index, v in enumerate(unq_nodes)}
rel2id = {r: index for index, r in enumerate(unq_rels)}

df_train["head_mapped"] = df_train["head"].map(node2id)  # .astype(int)
df_train["tail_mapped"] = df_train["tail"].map(node2id)  # .astype(int)
df_train["rel_mapped"] = df_train["rel"].map(rel2id)
df_train = df_train.dropna()


df_eval["head_mapped"] = df_eval["head"].map(node2id)  # .astype(int)
df_eval["tail_mapped"] = df_eval["tail"].map(node2id)  # .astype(int)
df_eval["rel_mapped"] = df_eval["rel"].map(rel2id)
df_eval = df_eval.dropna()

df_test["head_mapped"] = df_test["head"].map(node2id)  # .astype(int)
df_test["rel_mapped"] = df_test["rel"].map(rel2id)  # .astype(int)
df_test["tail_mapped"] = df_test["tail"].map(node2id)  # .astype(int)
df_test = df_test.dropna()


df_all["head_mapped"] = df_all["head"].map(node2id)  # .astype(int)
df_all["rel_mapped"] = df_all["rel"].map(rel2id)  # .astype(int)
df_all["tail_mapped"] = df_all["tail"].map(node2id)  # .astype(int)
df_all = df_all.dropna()


graph = dgl.graph(
    (df_all["head_mapped"].values.astype(int), df_all["tail_mapped"].values.astype(int))
)
graph.edata["rel"] = torch.tensor(df_all["rel_mapped"].values)


import random
import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # os.environ('PYTHONHASHSEED') = str(seed)


set_all_seeds(42)


epochs = 20
NUM_NEGATIVE = 10
embedding_dim = 200
lr = 1e-2
device = "cuda"
num_to_test = len(df_test)  # 30  # 30  # len(df_test)  # 20

train_mask = torch.zeros(len(df_all), dtype=int)
train_mask[: len(df_train)] = 1
val_mask = torch.zeros(len(df_all), dtype=int)
val_mask[len(df_train) : len(df_train) + len(df_eval)] = 1
test_mask = torch.zeros(len(df_all), dtype=int)
test_mask[len(df_train) + len(df_eval) : len(df_train) + len(df_eval) + num_to_test] = 1

train_graph = dgl.edge_subgraph(graph, train_mask.bool(), relabel_nodes=False)
val_graph = dgl.edge_subgraph(graph, val_mask.bool(), relabel_nodes=False)
test_graph = dgl.edge_subgraph(graph, test_mask.bool(), relabel_nodes=False)

model = TransE(graph.num_nodes(), len(rel2id), embedding_dim=embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    h, t = graph.edges()
    r = graph.edata["rel"]
    h_emb, r_emb, t_emb = model(h, r, t)
    neg_h, neg_t = generate_negative_samples(graph, h_emb.shape[0])
    neg_h_emb = model.entity_embeddings(neg_h)
    neg_t_emb = model.entity_embeddings(neg_t)

    loss = model.compute_loss(h_emb, r_emb, t_emb, neg_h_emb, neg_t_emb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")


nodesimilarity = model.entity_embeddings.weight @ model.entity_embeddings.weight.T

scores, ranks_simple = evaluate(model, test_graph)
print(f"\n Scores Generic: \n{scores}")


scores, rank_global = evaluate(model, test_graph, mode="adapt")
print(f"\n Scores Adapted: \n{scores}")
