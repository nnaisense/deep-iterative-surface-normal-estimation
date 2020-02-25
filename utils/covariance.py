import torch
from torch_scatter import scatter_add, scatter_mean


def compute_cov_matrices(pos, edge_idx):
    row, col = edge_idx
    (N, D), E = pos.size(), row.size(0)
    d = pos[col] - pos[row]
    # center
    centers = scatter_mean(d, row, dim=0, dim_size=N)
    d = d - centers[row]
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D))
    cov = scatter_add(cov, row, dim=0, dim_size=N)
    return cov


def compute_cov_matrices_dense(pos, dense_adj, row):
    col = dense_adj.view(-1)
    (N, D), E = pos.size(), (dense_adj.size(0) * dense_adj.size(1))
    d = (pos[col] - pos[row]).view(N, -1, 3)
    # center
    centers = d.mean(1)
    d = d - centers.view(-1, 1, 3)
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D)).view(N, -1, 3, 3)
    cov = cov.sum(1)
    return cov


def compute_weighted_cov_matrices(pos, weights, edge_idx):
    row, col = edge_idx
    (N, D), E = pos.size(), row.size(0)
    d = pos[col] - pos[row]
    # center
    weights_sum = scatter_add(weights, row, dim=0, dim_size=N)
    centers = scatter_add(d * weights.view(-1, 1), row, dim=0, dim_size=N) / weights_sum.view(-1, 1)

    d = d - centers[row]
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D))
    cov = cov * weights.view(-1, 1, 1)
    cov = scatter_add(cov, row, dim=0, dim_size=N)
    return cov


def compute_weighted_cov_matrices_dense(pos, weights, dense_adj, row):
    col = dense_adj.view(-1)
    (N, D), E = pos.size(), (dense_adj.size(0) * dense_adj.size(1))
    d = (pos[col] - pos[row]).view(N, -1, 3)
    # center
    weights_sum = weights.view(N, -1).sum(1)
    centers = (d * weights.view(N, -1, 1)).sum(1) / weights_sum.view(N, 1)
    d = d - centers.view(-1, 1, 3)
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D)).view(N, -1, 3, 3)
    cov = cov * weights.view(N, -1, 1, 1)
    cov = cov.sum(1)
    return cov
