import torch
import torch.nn as nn
from models.modules.resunet import ResUnet_DirectAttenMultiImg_Cond
from models.modules.parpoints_encoder import ParPoint_Encoder
from models.modules.PointEMB import PointEmbed
from models.modules.utils import StackedRandomGenerator
from models.modules.diffusion_sampler import edm_sampler
from models.modules.encoder import DiagonalGaussianDistribution
import numpy as np
class EDMLoss_MultiImgCond:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,use_par=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.use_par=use_par

    def __call__(self, net, data_batch, classifier_free=False):
        inputs = data_batch['input']
        image=data_batch['image']
        proj_mat=data_batch['proj_mat']
        valid_frames=data_batch['valid_frames']
        par_points=data_batch["par_points"]
        category_code=data_batch["category_code"]
        rnd_normal = torch.randn([inputs.shape[0], 1, 1, 1], device=inputs.device)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp() #B,1,1,1
        weight = (sigma ** 2 + self.sigma_data ** 2) / (self.sigma_data * sigma) ** 2
        y=inputs

        n = torch.randn_like(y) * sigma

        # if classifier_free and np.random.random()<0.5:
        #     net.par_feat=torch.zeros((inputs.shape[0],32,inputs.shape[2],inputs.shape[3])).float().to(inputs.device)
        if classifier_free and np.random.random()<0.5:
            image=torch.zeros_like(image).float().cuda()
        net.module.extract_img_feat(image)
        net.module.set_proj_matrix(proj_mat)
        net.module.set_valid_frames(valid_frames)
        net.module.set_category_code(category_code)
        if self.use_par:
            net.module.extract_point_feat(par_points)

        D_yn = net(y + n,sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss

class Triplane_Diff_MultiImgCond_EDM(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.diff_reso=opt['diff_reso']
        self.diff_dim=opt['output_channel']
        self.use_cat_embedding=opt['use_cat_embedding']
        self.use_fp16=False
        self.sigma_data=0.5
        self.sigma_max=float("inf")
        self.sigma_min=0
        self.use_par=opt['use_par']
        self.triplane_padding=opt['triplane_padding']
        self.block_type=opt['block_type']
        #self.use_bn=opt['use_bn']
        if opt['backbone']=="resunet_multiimg_direct_atten":
            self.denoise_model=ResUnet_DirectAttenMultiImg_Cond(channel=opt['input_channel'],
                                       output_channel=opt['output_channel'],use_par=opt['use_par'],par_channel=opt['par_channel'],
                                       img_in_channels=opt['img_in_channels'],vit_reso=opt['vit_reso'],triplane_padding=self.triplane_padding,
                                       norm=opt['norm'],use_cat_embedding=self.use_cat_embedding,block_type=self.block_type)
        else:
            raise NotImplementedError
        if opt['use_par']: #use partial point cloud as inputs
            par_emb_dim = opt['par_emb_dim']
            par_args = opt['par_point_encoder']
            self.point_embedder = PointEmbed(hidden_dim=par_emb_dim)
            self.par_points_encoder = ParPoint_Encoder(c_dim=par_args['plane_latent_dim'], dim=par_emb_dim,
                                                       plane_resolution=par_args['plane_reso'],
                                                       unet_kwargs=par_args['unet'])
        self.unflatten = torch.nn.Unflatten(1, (16, 16))
    def prepare_data(self,data_batch):
        #par_points = data_batch['par_points'].to(device, non_blocking=True)
        device=torch.device("cuda")
        means, logvars = data_batch['triplane_mean'].to(device, non_blocking=True), data_batch['triplane_logvar'].to(
            device, non_blocking=True)
        distribution = DiagonalGaussianDistribution(means, logvars)
        plane_feat = distribution.sample()

        image=data_batch["image"].to(device)
        proj_mat = data_batch['proj_mat'].to(device, non_blocking=True)
        valid_frames=data_batch["valid_frames"].to(device,non_blocking=True)
        par_points=data_batch["par_points"].to(device,non_blocking=True)
        category_code=data_batch["category_code"].to(device,non_blocking=True)
        input_dict = {"input": plane_feat.float(),
                      "image": image.float(),
                      "par_points":par_points.float(),
                      "proj_mat":proj_mat.float(),
                      "category_code":category_code.float(),
                      "valid_frames":valid_frames.float()}  # TODO: add image and proj matrix

        return input_dict

    def prepare_sample_data(self,data_batch):
        device=torch.device("cuda")
        image=data_batch['image'].to(device, non_blocking=True)
        proj_mat = data_batch['proj_mat'].to(device, non_blocking=True)
        valid_frames = data_batch["valid_frames"].to(device, non_blocking=True)
        par_points = data_batch["par_points"].to(device, non_blocking=True)
        category_code=data_batch["category_code"].to(device,non_blocking=True)
        sample_dict={
            "image":image.float(),
            "proj_mat":proj_mat.float(),
            "valid_frames":valid_frames.float(),
            "category_code":category_code.float(),
            "par_points":par_points.float(),
        }
        return sample_dict

    def prepare_eval_data(self,data_batch):
        device=torch.device("cuda")
        samples=data_batch["points"].to(device, non_blocking=True)
        labels=data_batch['labels'].to(device,non_blocking=True)

        eval_dict={
            "samples":samples,
            "labels":labels,
        }
        return eval_dict

    def extract_point_feat(self,par_points):
        par_emb=self.point_embedder(par_points)
        self.par_feat=self.par_points_encoder(par_points,par_emb)

    def extract_img_feat(self,image):
        self.image_emb=image

    def set_proj_matrix(self,proj_matrix):
        self.proj_matrix=proj_matrix

    def set_valid_frames(self,valid_frames):
        self.valid_frames=valid_frames

    def set_category_code(self,category_code):
        self.category_code=category_code

    def forward(self, x, sigma,force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1) #B,1,1,1
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4 #B,1,1,1, need to check how to add embedding into unet

        if self.use_par:
            F_x = self.denoise_model((c_in * x).to(dtype), c_noise.flatten(), self.image_emb, self.proj_matrix,
                                     self.valid_frames,self.category_code,self.par_feat)
        else:
            F_x = self.denoise_model((c_in * x).to(dtype), c_noise.flatten(),self.image_emb,self.proj_matrix,
                                     self.valid_frames,self.category_code)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(self, input_batch, batch_seeds=None,ret_all=False,num_steps=18):
        img_cond=input_batch['image']
        proj_mat=input_batch['proj_mat']
        valid_frames=input_batch["valid_frames"]
        category_code=input_batch["category_code"]
        if img_cond is not None:
            batch_size, device = img_cond.shape[0], img_cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        self.extract_img_feat(img_cond)
        self.set_proj_matrix(proj_mat)
        self.set_valid_frames(valid_frames)
        self.set_category_code(category_code)
        if self.use_par:
            par_points=input_batch["par_points"]
            self.extract_point_feat(par_points)
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.diff_dim, self.diff_reso*3,self.diff_reso], device=device)

        return edm_sampler(self, latents, randn_like=rnd.randn_like,ret_all=ret_all,sigma_min=0.002, sigma_max=80,num_steps=num_steps)


