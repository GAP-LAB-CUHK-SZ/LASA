import torch.nn as nn
import sys,os
sys.path.append("..")
import torch
from datasets import build_dataset
from configs.config_utils import CONFIG
from torch.utils.data import DataLoader
from models.modules import PointEmbed
from models.modules import ConvPointnet_Encoder,ConvPointnet_Decoder
import numpy as np

class TriplaneVAE(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.point_embedder=PointEmbed(hidden_dim=opt['point_emb_dim'])

        encoder_args=opt['encoder']
        decoder_args=opt['decoder']
        self.encoder=ConvPointnet_Encoder(c_dim=encoder_args['plane_latent_dim'],dim=opt['point_emb_dim'],latent_dim=encoder_args['latent_dim'],
                    plane_resolution=encoder_args['plane_reso'],unet_kwargs=encoder_args['unet'],unet=True,padding=opt['padding'])
        self.decoder=ConvPointnet_Decoder(latent_dim=decoder_args['latent_dim'],query_emb_dim=decoder_args['query_emb_dim'],
                                          hidden_dim=decoder_args['hidden_dim'],unet_kwargs=decoder_args['unet'],n_blocks=decoder_args['n_blocks'],
                                          plane_resolution=decoder_args['plane_reso'],padding=opt['padding'])

    def forward(self,p,query):
        '''
        :param p: surface points cloud of shape B,N,3
        :param query: sample points of shape B,N,3
        :return:
        '''
        point_emb=self.point_embedder(p)
        query_emb=self.point_embedder(query)
        kl,plane_feat,means,logvars=self.encoder(p,point_emb)
        if self.training:
            if np.random.random()<0.5:
                '''randomly sacle the triplane, and conduct triplane diffusion on 64x64x64 plane, promote robustness'''
                plane_feat=torch.nn.functional.interpolate(plane_feat,scale_factor=0.5,mode="bilinear")
                plane_feat=torch.nn.functional.interpolate(plane_feat,scale_factor=2,mode="bilinear")
        # if self.training:
        #     if np.random.random()<0.5:
        #         means = torch.nn.functional.interpolate(means, scale_factor=0.5, mode="bilinear")
        #         vars=torch.exp(logvars)
        #         vars = torch.nn.functional.interpolate(vars, scale_factor=0.5, mode="bilinear")
        #         new_logvars=torch.log(vars)
        #         posterior = DiagonalGaussianDistribution(means, new_logvars)
        #         plane_feat=posterior.sample()
        #         plane_feat=torch.nn.functional.interpolate(plane_feat,scale_factor=2,mode='bilinear')

        # mean_scale=torch.nn.functional.interpolate(means, scale_factor=0.5, mode="bilinear")
        # vars = torch.exp(logvars)
        # vars_scale = torch.nn.functional.interpolate(vars, scale_factor=0.5, mode="bilinear")/4
        # logvars_scale=torch.log(vars_scale)
        # scale_noise=torch.randn(mean_scale.shape).to(mean_scale.device)
        # plane_feat_scale2=mean_scale+torch.exp(0.5*logvars_scale)*scale_noise
        # plane_feat=torch.nn.functional.interpolate(plane_feat_scale2,scale_factor=2,mode='bilinear')
        o=self.decoder(plane_feat,query,query_emb)

        return {'logits':o,'kl':kl}


    def decode(self,plane_feature,query):
        query_embedding=self.point_embedder(query)
        o=self.decoder(plane_feature,query,query_embedding)

        return o

    def encode(self,p):
        point_emb = self.point_embedder(p)
        kl, plane_feat,mean,logvar = self.encoder(p, point_emb)
        '''p is point cloud of B,N,3'''
        return plane_feat,kl,mean,logvar

if __name__=="__main__":
    configs=CONFIG("../configs/train_triplane_vae_64.yaml")
    config=configs.config
    dataset_config=config['datasets']
    model_config=config["model"]
    dataset=build_dataset("train",dataset_config)
    dataset.__getitem__(0)
    dataloader=DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
    )
    net=TriplaneVAE(model_config).float().cuda()
    for idx,data_batch in enumerate(dataloader):
        if idx==1:
            break
        surface=data_batch['surface'].float().cuda()
        query=data_batch['points'].float().cuda()
        net(surface,query)


