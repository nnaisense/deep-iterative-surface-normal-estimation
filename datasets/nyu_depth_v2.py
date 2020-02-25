import os
import os.path as osp
import glob

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.read import read_txt_array
import h5py
import numpy as np

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

class NYUDepthV2_PC(InMemoryDataset):
    r""" Loads the NYU Depth V2 dataset as raw point clouds

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'


    def __init__(self,
                 root,
                 transform=None,
                 category='preprocessed',
                 pre_transform=None,
                 pre_filter=None):


        self.category = category
        super(NYUDepthV2_PC, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return ['nyu_depth_v2_labeled.mat']


    @property
    def processed_file_names(self):
        if self.category == 'raw':
            return 'point_clouds_raw.pt'
        elif self.category == 'preprocessed':
            return 'point_clouds_preprocessed.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        #extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        filename = self.raw_file_names[0]
        with h5py.File(osp.join(self.raw_dir, filename), 'r') as f:
            if self.category == 'raw':
                depthdata = np.array(f['rawDepths'])
            elif self.category == 'preprocessed':
                depthdata = np.array(f['depths'])
        data_list = []
        for i,image in enumerate(depthdata):
            image = np.swapaxes(image,0,1)
            x = np.arange(1, 641)
            y = np.arange(1, 481)
            xx, yy = np.meshgrid(x, y)
            X = (xx - cx_d) * image / fx_d
            Y = (yy - cy_d) * image / fy_d
            Z = image
            pos = np.stack([X,Y,Z], axis=-1)
            pos = torch.from_numpy(pos)
            pos = pos.reshape(-1, 3)
            mask = (pos[:,2] != 0.0)
            pos = pos[mask].float()
            y = np.stack([xx,yy], axis=-1).reshape(-1,2).astype(int)
            y = torch.from_numpy(y)
            y = y[mask]
            data = Data(pos=pos.reshape(-1, 3), y=y)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__,
                                            len(self), self.category)

