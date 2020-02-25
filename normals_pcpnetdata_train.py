import numpy as np
import torch
import argparse
import torch_geometric.transforms as T
from datasets.pcpnet_dataset import PCPNetDataset
from torch_geometric.data import DataLoader
from utils.radius import radius_graph
from torch_sym3eig import Sym3Eig
from networks.gnn import GNNFixedK
from utils.covariance import compute_cov_matrices_dense, compute_weighted_cov_matrices_dense


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='network_new_epoch{}.pt',
                    help='Model file name to store in trained_models/')
parser.add_argument('--dataset_path', type=str, default='data/pcpnet_data/', help='Path at which dataset is created')
parser.add_argument('--k_train', type=int, default=48, help='Neighborhood size for training [default: 48]')
parser.add_argument('--iterations', type=int, default=8, help='Number of iterations for testing [default: 4]')
FLAGS = parser.parse_args()


path = FLAGS.dataset_path
transform = T.Compose([T.NormalizeScale()])
train_dataset = PCPNetDataset(path, trainvaltest='train', category='Noisy', transform=transform)
val_all_dataset = PCPNetDataset(path, trainvaltest='val', category='NoisyAndVarDensity', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
val_all_loader = DataLoader(val_all_dataset, batch_size=1, shuffle=True)

# Dataloaders for all categories
test_nn_dataset = PCPNetDataset(path, trainvaltest='test', category='NoNoise', transform=transform)
test_nn_loader = DataLoader(test_nn_dataset, batch_size=1, pin_memory=True, num_workers=4)
test_ln_dataset = PCPNetDataset(path, trainvaltest='test', category='LowNoise', transform=transform)
test_ln_loader = DataLoader(test_ln_dataset, batch_size=1, pin_memory=True, num_workers=4)
test_mn_dataset = PCPNetDataset(path, trainvaltest='test', category='MedNoise', transform=transform)
test_mn_loader = DataLoader(test_mn_dataset, batch_size=1, pin_memory=True, num_workers=4)
test_hn_dataset = PCPNetDataset(path, trainvaltest='test', category='HighNoise', transform=transform)
test_hn_loader = DataLoader(test_hn_dataset, batch_size=1, pin_memory=True, num_workers=4)
test_vds_dataset = PCPNetDataset(path, trainvaltest='test', category='VarDensityStriped', transform=transform)
test_vds_loader = DataLoader(test_vds_dataset, batch_size=1, pin_memory=True, num_workers=4)
test_vdg_dataset = PCPNetDataset(path, trainvaltest='test', category='VarDensityGradient', transform=transform)
test_vdg_loader = DataLoader(test_vdg_dataset, batch_size=1, pin_memory=True, num_workers=4)

test_code_loader = DataLoader(test_nn_dataset[:2], batch_size=1)

# Normal estimation algorithm
# forward() corresponds to one iteration of Algorithm 1 in the paper
class NormalEstimation(torch.nn.Module):
    def __init__(self):
        super(NormalEstimation, self).__init__()
        self.stepWeights = GNNFixedK()
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, old_weights, pos, batch, normals, edge_idx_l, dense_l, stddev):
        # Re-weighting
        weights = self.stepWeights(pos, old_weights, normals, edge_idx_l, dense_l, stddev)  # , f=f)
        weights = self.dropout(weights)

        # Weighted Least-Squares
        cov = compute_weighted_cov_matrices_dense(pos, weights, dense_l, edge_idx_l[0])
        noise = (torch.rand(100, 3) - 0.5) * 1e-8
        cov = cov + torch.diag(noise).cuda()
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
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)


def train(epoch, size):
    model.train()

    if epoch == 151:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005


    loss_sum = [0.0 for _ in range(FLAGS.iterations)]
    loss_count = 0

    for i, data in enumerate(train_loader):
        pos, batch = data.pos, data.batch

        # Compute global statistics for normalization
        edge_idx_16, _ = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=16)
        row16, col16 = edge_idx_16
        cart16 = (pos[col16].cuda()-pos[row16].cuda())
        stddev = torch.sqrt((cart16**2).mean()).detach()

        # Compute KNN-graph indices for GNN
        edge_idx_l, dense_l = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=size)

        # Iteration 0 (PCA)
        cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        # mask = torch.isnan(eig_vec)
        # eig_vec[mask] = 0.0
        normals = eig_vec[:, :, 0].cuda()
        pos, batch = pos.detach().cuda(), batch.detach().cuda()
        edge_idx_c = edge_idx_l.cuda()
        old_weights = torch.ones_like(edge_idx_l[0]).float() / float(size)
        old_weights = old_weights.cuda() 

        normal_gt = data.x[:, 0:3].cuda()

        # Loop of Algorithm 1 in the paper
        for j in range(FLAGS.iterations):
            optimizer.zero_grad()
            normals, old_weights = model(old_weights.detach(), pos, batch, normals.detach(),
                                                                edge_idx_c, edge_idx_c[1].view(pos.size(0), -1), stddev)

            # Compute loss iteration j and optimize
            loss_orientation = torch.min(torch.sqrt(((normal_gt - normals) ** 2).sum(-1)),
                                         torch.sqrt(((normal_gt + normals) ** 2).sum(-1))).mean()
            loss_orientation.backward()

            loss_sum[j] += loss_orientation.detach().item()
            num_nan = 0
            for p in model.parameters():
                num_nan += torch.isnan(p.grad).sum()
                p.grad[torch.isnan(p.grad)] = 0.0
            if num_nan > 0:
                print('NUM_NAN:', num_nan)

            optimizer.step()
        loss_count += 1

    str = 'Epoch {}, Losses: '.format(epoch)
    for loss in loss_sum:
        str += '{:.7f}, '.format(loss / loss_count)
    print(str)


def test(loader, string, epoch, size):
    model.eval()
    num = 0
    error_wo_amb = [0.0 for _ in range(FLAGS.iterations+1)]
    error_wo_amb5000 = [0.0 for _ in range(FLAGS.iterations+1)]
    for i, data in enumerate(loader):
        pos, batch = data.pos, data.batch

        # Compute statistics for normalization
        edge_idx_16, _ = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=16)
        row16, col16 = edge_idx_16
        cart16 = (pos[col16].cuda()-pos[row16].cuda())
        stddev = torch.sqrt((cart16**2).mean()).detach()

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
        old_weights = old_weights  # .view(-1, 1).expand(-1, 3)

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

            # Compute error iteration j 
            '''
            # Test error all points
            abs_dot = torch.abs((normals.cpu() * normal_gt).sum(-1))
            abs_dot = torch.clamp(abs_dot, min=0.0, max=1.0)
            error_new_amb = torch.sqrt((torch.acos(abs_dot) ** 2).mean()).detach().item() * 180 / np.pi
            error_wo_amb[j] += error_new_amb
            abs_dot = 0
            '''

            # Indices of 5000 point subset stored in data.y (benchmark subset from PCPNet dataset/paper)
            abs_dot5000 = torch.abs((normals[data.y].cpu() * normal_gt[data.y]).sum(-1))
            abs_dot5000 = torch.clamp(abs_dot5000, min=0.0, max=1.0)
            error_new_amb5000 = torch.sqrt((torch.acos(abs_dot5000) ** 2).mean()).detach().item() * 180 / np.pi
            error_wo_amb5000[j + 1] += error_new_amb5000
            abs_dot5000 = 0

            normals = normals.detach()
            old_weights = old_weights.detach()
        num += 1
    error_wo_amb5000 = [x / num for x in error_wo_amb5000]

    str = 'Epoch: {:02d}, Unoriented Test E 5000: '.format(epoch)
    for i, error in enumerate(error_wo_amb5000):
        str += '{}: {:.4f}, '.format(i, error)
    print(str)

    error_wo_amb5000 = np.array([x for x in error_wo_amb5000])
    return error_wo_amb5000



def run():
    size = FLAGS.k_train
    print('Start training neighborhood size: {}'.format(size))
    best_avg_val = 70.0
    best_model_epoch = 0
    current_errors = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for epoch in range(1, 301):
        train(epoch, size)
        if epoch % 5 == 0:
            e = np.array([0.0 for _ in range(FLAGS.iterations+1)])
            c = test(test_nn_loader, 'NoNoise', epoch, size)
            e += c
            current_errors[0] = c[-1]
            c = test(test_ln_loader, 'LowNoise', epoch, size)
            e += c
            current_errors[1] = c[-1]
            c = test(test_mn_loader, 'MedNoise', epoch, size)
            e += c
            current_errors[2] = c[-1]
            c = test(test_hn_loader, 'HighNoise', epoch, size)
            e += c
            current_errors[3] = c[-1]
            c = test(test_vds_loader, 'VarDensityStriped', epoch, size)
            e += c
            current_errors[4] = c[-1]
            c = test(test_vdg_loader, 'VarDensityGradient', epoch, size)
            e += c
            current_errors[5] = c[-1]
            # test(test_all_loader, 'All', epoch)
            print('Average test error unoriented 5000:', e / 6.0)
            current_errors[6] = (e[-1]/6.0)
            print('Test on Val')
            v = test(val_all_loader, 'Validation All', epoch, size)
            if (v[-1]/6.0) < best_avg_val:
                best_avg_val = (v[-1]/6.0)
                best_model_epoch = epoch
                best_errors = current_errors.copy()
            print('Current best model: Epoch {}'.format(best_model_epoch), best_errors)
            torch.save(model.state_dict(), 'trained_models/{}'.format(FLAGS.model_name.format(epoch)))

run()
