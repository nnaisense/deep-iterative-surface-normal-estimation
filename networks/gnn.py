import torch
import sys

sys.path.append('..')
from torch_geometric.nn.inits import reset
from torch.nn import Sequential as S, Linear as L, ReLU
from utils.prf import compute_prf
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from utils.quaternion import QuatToMat


def scatter_softmax(x, index, size):
    exps = torch.exp(x)
    exps_sum = scatter_add(exps, index, dim=0, dim_size=size)
    return exps / exps_sum[index]


# Faster version for fixed sized k-neighborhoods
class GNNFixedK(torch.nn.Module):
    def __init__(self):
        super(GNNFixedK, self).__init__()

        self.layer1 = S(L(8, 32), ReLU(), L(32, 16))
        self.layerg = S(L(19, 32), ReLU(), L(32, 8))
        self.layer2 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg2 = S(L(16, 32), ReLU(), L(32, 8))
        self.layer3 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg3 = S(L(16, 32), ReLU(), L(32, 12))
        self.layer4 = S(L(27, 64), ReLU(), L(64, 1))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.layer1)
        reset(self.layerg)
        reset(self.layer2)
        reset(self.layerg2)
        reset(self.layer3)
        reset(self.layerg3)
        reset(self.layer4)

    def forward(self, pos, old_weights, normals, edge_index, dense_l, stddev, f=None):
        """"""
        N = pos.size(0)
        K = dense_l.size(1)
        E = edge_index.size(1)

        rows, cols = edge_index
        cart = pos[cols] - pos[rows]
        scale = 0.2 / stddev
        cart = cart * scale
        ppf = compute_prf(pos, normals, edge_index, scale=scale)

        x = torch.cat([cart, old_weights.view(-1, 1), ppf], dim=-1)
        x = self.layer1(x)
        x = x.view(N, K, -1)
        global_x = x.mean(1)
        if f is not None:
            global_x = torch.cat([global_x, normals.view(-1, 3), f.view(-1, 32)], dim=-1)
        else:
            global_x = torch.cat([global_x, normals.view(-1, 3)], dim=-1)
        x_g = self.layerg(global_x)
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)
        x = self.layer2(x)
        x = x.view(N, K, -1)
        global_x = x.mean(1)
        x_g = self.layerg2(global_x)
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)
        x = self.layer3(x)
        x = x.view(N, K, -1)
        global_x = x.mean(1)
        x_g = self.layerg3(global_x)
        quat = x_g[:, :4]
        quat = quat / (quat.norm(p=2, dim=-1) + 1e-8).view(-1, 1)
        mat = QuatToMat.apply(quat).view(-1, 3, 3)

        # Kernel application
        x_g = x_g[:, 4:]
        rot_cart = torch.matmul(mat.view(-1, 3, 3)[rows], cart.view(-1, 3, 1)).view(-1, 3)
        x = torch.cat([x.view(E, -1), x_g[rows], rot_cart], dim=1)
        x = self.layer4(x)
        x = x.view(N, K)
        weights = F.softmax(x, 1)
        return weights


# Version for variable-sized neighborhoods (k and max r), using scatter+gather
class GNNVariableK(torch.nn.Module):
    def __init__(self):
        super(GNNVariableK, self).__init__()

        self.layer1 = S(L(8, 32), ReLU(), L(32, 16))
        self.layerg = S(L(19, 32), ReLU(), L(32, 8))
        self.layer2 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg2 = S(L(16, 32), ReLU(), L(32, 8))
        self.layer3 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg3 = S(L(16, 32), ReLU(), L(32, 12))
        self.layer4 = S(L(27, 64), ReLU(), L(64, 1))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.layer1)
        reset(self.layerg)
        reset(self.layer2)
        reset(self.layerg2)
        reset(self.layer3)
        reset(self.layerg3)
        reset(self.layer4)

    def forward(self, pos, old_weights, normals, edge_index, stddev, f=None):
        """"""
        N = pos.size(0)
        E = edge_index.size(1)

        rows, cols = edge_index
        cart = pos[cols] - pos[rows]
        scale = 0.2 / stddev
        cart = cart * scale
        ppf = compute_prf(pos, normals, edge_index, scale=scale)

        x = torch.cat([cart, old_weights.view(-1, 1), ppf], dim=-1)
        x = self.layer1(x)
        global_x = scatter_mean(x, rows, 0, dim_size=N)
        if f is not None:
            global_x = torch.cat([global_x, normals.view(-1, 3), f.view(-1, 32)], dim=-1)
        else:
            global_x = torch.cat([global_x, normals.view(-1, 3)], dim=-1)
        x_g = self.layerg(global_x)
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)
        x = self.layer2(x)
        global_x = scatter_mean(x, rows, 0, dim_size=N)
        x_g = self.layerg2(global_x)
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)
        x = self.layer3(x)
        global_x = scatter_mean(x, rows, 0, dim_size=N)
        x_g = self.layerg3(global_x)
        quat = x_g[:, :4]
        quat = quat / (quat.norm(p=2, dim=-1) + 1e-8).view(-1, 1)
        mat = QuatToMat.apply(quat).view(-1, 3, 3)

        # Kernel application
        x_g = x_g[:, 4:]
        rot_cart = torch.matmul(mat.view(-1, 3, 3)[rows], cart.view(-1, 3, 1)).view(-1, 3)
        x = torch.cat([x.view(E, -1), x_g[rows], rot_cart], dim=1)
        x = self.layer4(x)
        weights = scatter_softmax(x, rows, size=N)
        return weights
