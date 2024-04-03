import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from .unet import UNet
from .resnet_block import ResnetBlockFC
import numpy as np

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2,3])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class ConvPointnet_Encoder(nn.Module):
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

    def __init__(self, c_dim=128, dim=3, hidden_dim=128,latent_dim=32, scatter_type='max',
                 unet=False, unet_kwargs=None,
                 plane_resolution=None, plane_type=['xz', 'xy', 'yz'], padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(unet_kwargs['output_dim'], in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean

        self.mean_fc = nn.Conv2d(unet_kwargs['output_dim'], latent_dim,kernel_size=1)
        self.logvar_fc = nn.Conv2d(unet_kwargs['output_dim'], latent_dim,kernel_size=1)

    # takes in "p": point cloud and "query": sdf_xyz
    # sample plane features for unlabeled_query as well
    def forward(self, p,point_emb):  # , query2):
        batch_size, T, D = p.size()
        #print('origin',torch.amin(p[0],dim=0),torch.amax(p[0],dim=0))
        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)
        net = self.fc_pos(point_emb)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        #print(c.shape)

        fea = {}
        plane_feat_sum = 0
        # second_sum = 0
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c,
                                                     plane='xz')  # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        cat_feature = torch.cat([fea['xz'], fea['xy'], fea['yz']],
                                dim=2)  # concat at row dimension
        #print(cat_feature.shape)
        plane_feat=self.unet(cat_feature)

        mean=self.mean_fc(plane_feat)
        logvar=self.logvar_fc(plane_feat)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x, mean, logvar


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

    # xy is the normalized coordinates of the point cloud of each plane
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = self.coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)
        #print(fea_plane.shape)

        return fea_plane

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True,
                                     mode='bilinear').squeeze(-1)
        return sampled_feat



