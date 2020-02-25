import torch


def cangle(vec1, vec2):
    n = vec1.norm(p=2, dim=-1)*vec2.norm(p=2, dim=-1)
    mask = (n < 1e-8).float()
    cang = (1-mask)*(vec1*vec2).sum(-1)/(n+mask)
    return cang


def compute_prf(pos, normals, edge_idx, scale=10.0):
    row, col = edge_idx
    d = pos[col] - pos[row]
    normals1 = normals[row]
    normals2 = normals[col]
    ppf = torch.stack([cangle(normals1, d), cangle(normals2, d),
                       cangle(normals1, normals2), torch.sqrt((d**2).sum(-1))*scale], dim=-1)
    return ppf
