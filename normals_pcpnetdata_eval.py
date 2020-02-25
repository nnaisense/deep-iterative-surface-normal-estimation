import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
import argparse
from datasets.pcpnet_dataset import PCPNetDataset
from torch_geometric.data import DataLoader
from utils.radius import radius_graph

from torch_sym3eig import Sym3Eig
import os
from networks.gnn import GNNFixedK
from utils.covariance import compute_cov_matrices_dense, compute_weighted_cov_matrices_dense

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default=None, help='Path were results (normals) are stored')
parser.add_argument('--model_name', default='network_k64.pt', help='Model file from trained_models/ to use')
parser.add_argument('--dataset_path', type=str, default='data/pcpnet_data/', help='Path at which dataset is created')
parser.add_argument('--k_test', type=int, default=64, help='Neighborhood size for eval [default: 64]')
parser.add_argument('--iterations', type=int, default=4, help='Number of iterations for testing [default: 4]')
FLAGS = parser.parse_args()

if FLAGS.results_path is not None:
    if not os.path.exists(FLAGS.results_path):
        os.makedirs(FLAGS.results_path)

path = FLAGS.dataset_path

transform = T.Compose([T.NormalizeScale()])
train_dataset = PCPNetDataset(path, trainvaltest='train', category='Noisy', transform=transform)
test_all_dataset = PCPNetDataset(path, trainvaltest='test', category='All')
val_dataset = PCPNetDataset(path, trainvaltest='val', category='NoisyAndVarDensity', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
test_all_loader = DataLoader(test_all_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

test_nn_dataset = PCPNetDataset(path, trainvaltest='test', category='NoNoise', transform=transform)
test_nn_loader = DataLoader(test_nn_dataset, batch_size=1, pin_memory=True, shuffle=False)
test_ln_dataset = PCPNetDataset(path, trainvaltest='test', category='LowNoise', transform=transform)
test_ln_loader = DataLoader(test_ln_dataset, batch_size=1, pin_memory=True, shuffle=False)
test_mn_dataset = PCPNetDataset(path, trainvaltest='test', category='MedNoise', transform=transform)
test_mn_loader = DataLoader(test_mn_dataset, batch_size=1, pin_memory=True,  shuffle=False)
test_hn_dataset = PCPNetDataset(path, trainvaltest='test', category='HighNoise', transform=transform)
test_hn_loader = DataLoader(test_hn_dataset, batch_size=1, pin_memory=True,  shuffle=False)
test_vds_dataset = PCPNetDataset(path, trainvaltest='test', category='VarDensityStriped', transform=transform)
test_vds_loader = DataLoader(test_vds_dataset, batch_size=1, pin_memory=True,  shuffle=False)
test_vdg_dataset = PCPNetDataset(path, trainvaltest='test', category='VarDensityGradient', transform=transform)
test_vdg_loader = DataLoader(test_vdg_dataset, batch_size=1, pin_memory=True,  shuffle=False)

test_code_loader = DataLoader(test_nn_dataset[:2], batch_size=1)

category_files_test = ['testset_no_noise.txt',
        'testset_low_noise.txt', 'testset_med_noise.txt',
        'testset_high_noise.txt', 'testset_vardensity_striped.txt',
        'testset_vardensity_gradient.txt']


def save_normals(normals, test_set, example):
    category_file = category_files_test[test_set]
    file_path = osp.join(path, 'raw', category_file)
    with open(file_path, "r") as f:
        filenames = f.read().split('\n')[:-1]
    file = filenames[example]
    out_path = osp.join(FLAGS.results_path, file+'.normals')
    normals = normals.cpu().numpy()
    np.savetxt(out_path, normals, delimiter=' ')


# Normal estimation algorithm
# forward() corresponds to one iteration of Algorithm 1 in the paper
class NormalEstimation(torch.nn.Module):
    def __init__(self):
        super(NormalEstimation, self).__init__()
        self.stepWeights = GNNFixedK()

    def forward(self, old_weights, pos, batch, normals, edge_idx_l, dense_l, stddev):
        # Re-weighting
        weights = self.stepWeights(pos, old_weights, normals, edge_idx_l, dense_l, stddev)  # , f=f)

        # Weighted Least-Squares
        cov = compute_weighted_cov_matrices_dense(pos, weights, dense_l, edge_idx_l[0])
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        normals = eig_vec[:, :, 0]

        # Not necessary for PCPNetDataset but might be for other datasets with underdefined neighborhoods
        # mask = torch.isnan(normals)
        # normals[mask] = 0.0

        return normals, weights


device = torch.device('cuda')
model = NormalEstimation().to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('num_params:', params)


def test(loader, string, test_set, size):
    model.eval()
    print('Starting eval: {}, k_test = {}, Iterations: {} '.format(string, size, FLAGS.iterations))
    num = 0
    error_wo_amb5000 = [0.0 for _ in range(FLAGS.iterations+1)]
    with torch.no_grad():
        for i, data in enumerate(loader):
            pos, batch = data.pos, data.batch

            # Compute statistics for normalization
            edge_idx_16, _ = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=16)
            row16, col16 = edge_idx_16
            cart16 = (pos[col16].cuda() - pos[row16].cuda())
            stddev = torch.sqrt((cart16 ** 2).mean()).detach().item()

            # Compute KNN-graph indices for GNN
            edge_idx_l, dense_l = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=size)

            # Iteration 0 (PCA)
            cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()
            eig_val, eig_vec = Sym3Eig.apply(cov)
            _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
            eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
            # mask = torch.isnan(eig_vec)
            # eig_vec[mask] = 0.0 
            normals = eig_vec[:, :, 0]
            edge_idx_c = edge_idx_l.cuda()
            pos, batch = pos.detach().cuda(), batch.detach().cuda()
            old_weights = torch.ones_like(edge_idx_c[0]).float() / float(size)

            # Compute error iteration 0 (PCA), 
            # Indices of 5000 point subset stored in data.y (benchmark subset from PCPNet dataset/paper)
            normal_gt = data.x[:, 0:3]
            abs_dot5000 = torch.abs((normals[data.y].cpu() * normal_gt[data.y]).sum(-1))
            abs_dot5000 = torch.clamp(abs_dot5000, min=0.0, max=1.0)
            error_new_amb5000 = torch.sqrt((torch.acos(abs_dot5000) ** 2).mean()).detach().item() * 180 / np.pi
            error_wo_amb5000[0] += error_new_amb5000
            abs_dot5000 = 0

            # Loop of Algorithm 1 in the paper
            for j in range(FLAGS.iterations):
                normals, old_weights = model(old_weights.detach(), pos, batch, normals.detach(),
                                                                     edge_idx_c, edge_idx_c[1].view(pos.size(0), -1), stddev)

                # Compute error iteration j, 
                # Indices of 5000 point subset stored in data.y (benchmark subset from PCPNet dataset/paper)
                abs_dot5000 = torch.abs((normals[data.y].cpu() * normal_gt[data.y]).sum(-1))
                abs_dot5000 = torch.clamp(abs_dot5000, min=0.0, max=1.0)
                error_new_amb5000 = torch.sqrt((torch.acos(abs_dot5000) ** 2).mean()).detach().item() * 180 / np.pi
                error_wo_amb5000[j + 1] += error_new_amb5000
                abs_dot5000 = 0


                normals = normals.detach()
                old_weights = old_weights.detach()
            
            num += 1
            if (i+1) % 5 == 0:
                print('{}/{} point clouds done'.format(i+1, len(loader)))
            if FLAGS.results_path is not None:
                save_normals(normals, test_set, i)

        error_wo_amb5000 = [x / num for x in error_wo_amb5000]
        print('{} Unoriented Normal Angle RMSE: PCA (0 Iterations): {:.4f}, {} Iterations: {:.4f}'.format(
              string,
              error_wo_amb5000[0], FLAGS.iterations, error_wo_amb5000[-1]))
        error_wo_amb5000 = np.array([x for x in error_wo_amb5000])
    return error_wo_amb5000



def run():
    size = FLAGS.k_test
    e = np.array([0.0 for _ in range(FLAGS.iterations+1)])
    e += test(test_nn_loader, 'NoNoise', 0, size)
    e += test(test_ln_loader, 'LowNoise', 1, size)
    e += test(test_mn_loader, 'MedNoise', 2, size)
    e += test(test_hn_loader, 'HighNoise', 3, size)
    e += test(test_vds_loader, 'VarDensityStriped', 4, size)
    e += test(test_vdg_loader, 'VarDensityGradient', 5, size)
    print('Average test error: PCA (0 Iterations), {} Iterations: {}', e[0] / 6.0, FLAGS.iterations, e[-1] / 6.0)


model.load_state_dict(torch.load('trained_models/{}'.format(FLAGS.model_name)))
run()
