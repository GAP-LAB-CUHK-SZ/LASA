import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from .unet import UNet
from .resnet_block import ResnetBlockFC
import numpy as np

class ConvPointnet_Decoder(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, latent_dim=32,query_emb_dim=51,hidden_dim=128, unet_kwargs=None,
                 plane_resolution=None, plane_type=['xz', 'xy', 'yz'], padding=0.1, n_blocks=5):
        super().__init__()

        self.latent_dim=32
        self.actvn = nn.ReLU()

        self.unet = UNet(unet_kwargs['output_dim'], in_channels=latent_dim, **unet_kwargs)

        self.fc_c=nn.ModuleList
        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding
        self.n_blocks=n_blocks

        self.fc_c = nn.ModuleList([
            nn.Linear(latent_dim*3, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_p=nn.Linear(query_emb_dim,hidden_dim)
        self.fc_out=nn.Linear(hidden_dim,1)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

    def forward(self, plane_features,query,query_emb):  # , query2):
        plane_feature=self.unet(plane_features)
        H,W=plane_feature.shape[2:4]
        xz_feat,xy_feat,yz_feat=torch.split(plane_feature,dim=2,split_size_or_sections=H//3)
        xz_sample_feat=self.sample_plane_feature(query,xz_feat,'xz')
        xy_sample_feat=self.sample_plane_feature(query,xy_feat,'xy')
        yz_sample_feat=self.sample_plane_feature(query,yz_feat,'yz')

        sample_feat=torch.cat([xz_sample_feat,xy_sample_feat,yz_sample_feat],dim=1)
        sample_feat=sample_feat.transpose(1,2)

        net=self.fc_p(query_emb)
        for i in range(self.n_blocks):
            net=net+self.fc_c[i](sample_feat)
            net=self.blocks[i](net)
        out=self.fc_out(self.actvn(net)).squeeze(-1)
        return out


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]
        #print("origin",torch.amin(xy), torch.amax(xy))
        xy=xy/2 #xy is originally -1 ~ 1
        xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
        xy_new = xy_new + 0.5  # range (0, 1)
        #print("scale",torch.amin(xy_new),torch.amax(xy_new))

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            x (tensor): coordinate
            reso (int): defined resolution
            coord_type (str): coordinate type
        '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :]
        return index

    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat



