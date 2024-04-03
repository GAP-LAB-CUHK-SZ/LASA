import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from external.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from .utils import zero_module
from .Positional_Embedding import PositionalEmbedding

class Pointnet2Encoder(nn.Module):
    def __init__(self,input_feature_dim=0,npoints=[2048,1024,512,256],radius=[0.2,0.4,0.6,1.2],nsample=[64,32,16,8]):
        super().__init__()
        self.sa1 = PointnetSAModuleVotes(
            npoint=npoints[0],
            radius=radius[0],
            nsample=nsample[0],
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=npoints[1],
            radius=radius[1],
            nsample=nsample[1],
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=npoints[2],
            radius=radius[2],
            nsample=nsample[2],
            mlp=[256, 256, 256, 512],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=npoints[3],
            radius=radius[3],
            nsample=nsample[3],
            mlp=[512, 512, 512, 512],
            use_xyz=True,
            normalize_xyz=True
        )
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features
    def forward(self,pointcloud,end_points=None):
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        end_points['org_xyz'] = xyz
        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz1, features1, _ = self.sa1(xyz, features)
        end_points['sa1_xyz'] = xyz1
        end_points['sa1_features'] = features1

        xyz2, features2, _ = self.sa2(xyz1, features1)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_xyz'] = xyz2
        end_points['sa2_features'] = features2

        xyz3, features3, _ = self.sa3(xyz2, features2)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz3
        end_points['sa3_features'] = features3
        #print(xyz3.shape,features3.shape)
        xyz4, features4, _ = self.sa4(xyz3, features3)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz4
        end_points['sa4_features'] = features4
        #print(xyz4.shape,features4.shape)
        return end_points



class PointUNet(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self):
        super().__init__()

        self.noisy_encoder=Pointnet2Encoder()
        self.cond_encoder=Pointnet2Encoder()
        self.fp1_cross = PointnetFPModule(mlp=[512 + 512, 512, 512])
        self.fp1 = PointnetFPModule(mlp=[512 + 512, 512, 512])
        #self.fp1 = PointnetFPModule(mlp=[512 + 512, 512, 512])
        self.fp2_cross = PointnetFPModule(mlp=[512 + 512, 512, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 512, 256])
        #self.fp2=PointnetFPModule(mlp=[512 + 256, 512, 256])
        self.fp3_cross= PointnetFPModule(mlp=[256 + 256, 256, 128])
        self.fp3 = PointnetFPModule(mlp=[128 + 128, 256, 128])
        #self.fp3 = PointnetFPModule(mlp=[256 + 128, 256, 128])
        self.fp4_cross=PointnetFPModule(mlp=[128+128, 128, 128])
        self.fp4 = PointnetFPModule(mlp=[128, 128, 128])
        #self.fp4 = PointnetFPModule(mlp=[128, 128, 128])

        self.output_layer=nn.Sequential(
            nn.LayerNorm(128),
            zero_module(nn.Linear(in_features=128,out_features=3,bias=False))
        )
        self.t_emb_layer = PositionalEmbedding(256)
        self.map_layer0 = nn.Linear(in_features=256, out_features=512)
        self.map_layer1 = nn.Linear(in_features=512, out_features=512)

    def forward(self, noise_points, t,cond_points):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        t_emb = self.t_emb_layer(t)
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))#B,512
        t_emb = t_emb[:, :, None] #B,512,K
        noise_end_points=self.noisy_encoder(noise_points)
        cond=self.cond_encoder(cond_points)
        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1_cross(noise_end_points['sa4_xyz'],cond['sa4_xyz'],noise_end_points['sa4_features']+t_emb,
                                  cond['sa4_features'])
        features = self.fp1(noise_end_points['sa3_xyz'], noise_end_points['sa4_xyz'], noise_end_points['sa3_features'],
                            features)
        features = self.fp2_cross(noise_end_points['sa3_xyz'],cond['sa3_xyz'],features,
                                  cond["sa3_features"])
        features = self.fp2(noise_end_points['sa2_xyz'], noise_end_points['sa3_xyz'], noise_end_points['sa2_features'],
                            features)
        features = self.fp3_cross(noise_end_points['sa2_xyz'],cond['sa2_xyz'],features,
                                  cond['sa2_features'])
        features = self.fp3(noise_end_points['sa1_xyz'],noise_end_points['sa2_xyz'],noise_end_points['sa1_features'],features)
        features = self.fp4_cross(noise_end_points['sa1_xyz'],cond['sa1_xyz'],features,
                                  cond['sa1_features'])
        features = self.fp4(noise_end_points['org_xyz'], noise_end_points['sa1_xyz'], None, features)
        features=features.transpose(1,2)

        # features = self.fp1_cross(noise_end_points['sa4_xyz'], cond_end_points['sa4_xyz'],
        #                           noise_end_points['sa4_features']+t_emb, cond_end_points['sa4_features'])
        # features = self.fp1(noise_end_points['sa3_xyz'].clone(), noise_end_points['sa4_xyz'].clone(), noise_end_points['sa3_features'],
        #                     features)
        # features = self.fp2(noise_end_points['sa2_xyz'], noise_end_points['sa3_xyz'], noise_end_points['sa2_features'],
        #                     features)
        # features = self.fp3(noise_end_points['sa1_xyz'],noise_end_points['sa2_xyz'],noise_end_points['sa1_features'],features)
        # features = self.fp4(noise_end_points['org_xyz'], noise_end_points['sa1_xyz'], None, features)
        # features = features.transpose(1,2)
        output_points=self.output_layer(features)

        return output_points


if __name__ == '__main__':
    net=PointUNet().cuda().float()
    net=net.eval()
    noise_points=torch.randn(16,4096,3).cuda().float()
    cond_points=torch.randn(16,4096,3).cuda().float()
    t=torch.randn(16).cuda().float()
    cond_encoder=Pointnet2Encoder().cuda().float()

    out = net(noise_points,cond_points)
    print(out.shape)