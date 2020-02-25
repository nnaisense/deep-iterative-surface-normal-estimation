import os
import numpy as np
import torch
import argparse
import torch_geometric.transforms as T
from datasets.nyu_depth_v2 import NYUDepthV2_PC
from torch_geometric.data import DataLoader
from utils.radius import radius_graph

from torch_sym3eig import Sym3Eig

from networks.gnn import GNNFixedK
from utils.covariance import compute_cov_matrices_dense, compute_weighted_cov_matrices_dense
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default='nyu_out/', help='Path were results (normals) are stored')
parser.add_argument('--model_name', default='network_k64.pt', help='Model file from trained_models/ to use')
parser.add_argument('--dataset_path', type=str, default='data/nyudepthv2/', help='Path at which dataset is created')
parser.add_argument('--k_test', type=int, default=64, help='Neighborhood size for eval [default: 64]')
parser.add_argument('--iterations', type=int, default=4, help='Number of iterations for testing [default: 4]')
FLAGS = parser.parse_args()

path = FLAGS.dataset_path
transform = T.Compose([T.NormalizeScale()])
dataset = NYUDepthV2_PC(path, transform=None)
loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

if not os.path.exists(FLAGS.results_path):
    os.makedirs(FLAGS.results_path)

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
        noise = (torch.rand(100, 3) - 0.5) * 1e-8
        cov = cov + torch.diag(noise).cuda()
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        normals = eig_vec[:, :, 0]

        # For underdefined neighborhoods
        mask = torch.isnan(normals)
        normals[mask] = 0.0  

        return normals, weights


device = torch.device('cuda')
model = NormalEstimation().to(device)

# Dataset constants
# Rotation
R = -np.array([ 9.9997798940829263e-01, 5.0518419386157446e-03,
   4.3011152014118693e-03, -5.0359919480810989e-03,
   9.9998051861143999e-01, -3.6879781309514218e-03,
   -4.3196624923060242e-03, 3.6662365748484798e-03,
   9.9998394948385538e-01 ]).reshape(3,3)

R = np.transpose(R,axes=[0,1])

# 3D Translation
t = np.array([2.5031875059141302e-02, 6.6238747008330102e-04, -2.9342312935846411e-04])
R = torch.from_numpy(R).float()
t = torch.from_numpy(t).float()


def test(loader, size):
    global R, t
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            original_pos = data.pos
            data = transform(data)
            pos, batch = data.pos, data.batch
            edge_idx_16, _ = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=16)
            row16, col16 = edge_idx_16
            cart16 = (pos[col16].cuda() - pos[row16].cuda())
            stddev = torch.sqrt((cart16 ** 2).mean()).detach()

            # split in 4
            pos1 = pos.view(480, 640, 3)[:260, :340].float()
            mask1 = torch.zeros_like(pos1[:, :, 0])
            mask1[:240, :320] = 1.0

            pos2 = pos.view(480, 640, 3)[:260, 300:].float()
            mask2 = torch.zeros_like(pos2[:, :, 0])
            mask2[:240, 20:] = 1.0

            pos3 = pos.view(480, 640, 3)[220:, :340].float()
            mask3 = torch.zeros_like(pos3[:, :, 0])
            mask3[20:, :320] = 1.0

            pos4 = pos.view(480, 640, 3)[220:, 300:].float()
            mask4 = torch.zeros_like(pos4[:, :, 0])
            mask4[20:, 20:] = 1.0

            batch = torch.zeros_like(pos1[:, :, 0])

            examples = [(pos1.contiguous().view(-1, 3), mask1, batch.view(-1)),
                        (pos2.contiguous().view(-1, 3), mask2, batch.view(-1)),
                        (pos3.contiguous().view(-1, 3), mask3, batch.view(-1)),
                        (pos4.contiguous().view(-1, 3), mask4, batch.view(-1))]

            normals_list = []
            for (pos, mask_part, batch) in examples:
                # print(pos.size(), batch.size(), mask_part.size())
                edge_idx_l, dense_l = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=size)
                cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()

                eig_val, eig_vec = Sym3Eig.apply(cov)
                _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
                eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
                mask = torch.isnan(eig_vec)
                eig_vec[mask] = 0.0  # For underdefined neighborhoods
                normals = eig_vec[:, :, 0]
                edge_idx_c = edge_idx_l.cuda()
                pos, batch = pos.detach().cuda(), batch.detach().cuda()
                old_weights = torch.ones_like(edge_idx_c[0]).float() / float(size)
                old_weights = old_weights  # .view(-1, 1).expand(-1, 3)

                # Loop of Algorithm 1 in the paper
                for j in range(FLAGS.iterations):
                    normals, old_weights = model(old_weights.detach(), pos, batch, normals.detach(),
                                                 edge_idx_c, edge_idx_c[1].view(pos.size(0), -1), stddev)
                    normals = normals.detach()
                    old_weights = old_weights.detach()

                mask_part = (mask_part == 1.0).view(-1)
                normals = normals[mask_part]
                normals = normals.view(240, 320, 3)
                normals_list.append(normals)

            row1 = torch.cat([normals_list[0], normals_list[1]], dim=1)
            row2 = torch.cat([normals_list[2], normals_list[3]], dim=1)
            result = torch.cat([row1, row2], dim=0).contiguous()

            pos = original_pos.float()
            # Coord Transform
            rot = torch.matmul(R.view(1, 3, 3), pos.view(-1, 3, 1)).view(-1, 3)
            pos = rot + t.view(1, 3)

            # flip to camera
            pos_dir = (torch.zeros_like(pos) - pos).cuda()
            sign = torch.sign((result * pos_dir.view(480, 640, 3)).sum(-1))
            sign[sign == 0.0] = 1.0
            result = result * sign.view(480, 640, 1)

            row, col = edge_idx_16
            # print(col.view(-1,16).size())
            sign_dist_to_neighbors = (result.view(-1, 3)[col].view(-1, 17, 3)[:, 1:, :] * result.view(-1, 1, 3)).sum(
                -1).mean(-1)
            # flip_mask = torch.stack([(sign_dist_to_neighbors<-0.6), (torch.abs((result.view(-1,3)*pos_dir).sum(-1)) < 0.3)], dim=1)
            # flip_mask = -flip_mask.all(-1).float()
            flip_mask = -(sign_dist_to_neighbors < 0.0).float()
            flip_mask[flip_mask == 0] = 1.0
            result = result * flip_mask.view(480, 640, 1)

            # result[:,2] = -result[:,2]

            '''
            # pc
            pcd = open3d.PointCloud()
            pos = pos.view(-1, 3).cpu().numpy()
            result = result.view(-1,3).cpu().numpy()
            pcd.points = open3d.Vector3dVector(pos)
            #pcd.normals = open3d.Vector3dVector(result)
            result = normals_to_rgb(result)
            pcd.colors = open3d.Vector3dVector(result)
            mesh_frame = open3d.create_mesh_coordinate_frame(size=0.6, origin=[0, 0, 0])
            open3d.draw_geometries([pcd, mesh_frame])
            '''
            # image
            result = (result + 1.0) * 0.5
            result = result.cpu().numpy()
            result = np.clip(result, a_min=0.0, a_max=1.0)
            image = Image.fromarray(np.uint8(result * 255))
            image.save(FLAGS.results_path + '/ex{}.png'.format(i))

            print('Image stored')


def run(n):
    test(loader, n)


model.load_state_dict(torch.load('trained_models/' + FLAGS.model_name))
run(FLAGS.k_test)
