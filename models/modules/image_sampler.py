import sys
sys.path.append('../..')
import torch
import torch.nn as nn
import math
from models.modules.unet import RollOut_Conv
from einops import rearrange, reduce
MB =1024.0*1024.0
def mask_kernel(x, sigma=1):
    return torch.abs(x) < sigma #if the distance is smaller than the kernel size, return True

def mask_kernel_close_false(x, sigma=1):
    return torch.abs(x) > sigma #if the distance is smaller than the kernel size, return False

class Image_Local_Sampler(nn.Module):
    def __init__(self,reso,padding=0.1,in_channels=1280,out_channels=512):
        super().__init__()
        self.triplane_reso=reso
        self.padding=padding
        self.get_triplane_coord()
        self.img_proj=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
    def get_triplane_coord(self):
        '''xz plane firstly, z is at the '''
        x=torch.arange(self.triplane_reso)
        z=torch.arange(self.triplane_reso)
        X,Z=torch.meshgrid(x,z,indexing='xy')
        xz_coords=torch.cat([X[:,:,None],torch.ones_like(X[:,:,None])*(self.triplane_reso-1)/2,Z[:,:,None]],dim=-1) #in xyz order

        '''xy plane'''
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        xy_coords = torch.cat([X[:, :, None],  Y[:, :, None],torch.ones_like(X[:, :, None])*(self.triplane_reso-1)/2], dim=-1)  # in xyz order

        '''yz plane'''
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)
        Y,Z = torch.meshgrid(y,z,indexing='xy')
        yz_coords= torch.cat([torch.ones_like(Y[:, :, None])*(self.triplane_reso-1)/2,Y[:,:,None],Z[:,:,None]], dim=-1)

        triplane_coords=torch.cat([xz_coords,xy_coords,yz_coords],dim=0)
        triplane_coords=triplane_coords/(self.triplane_reso-1)
        triplane_coords=(triplane_coords-0.5)*2*(1 + self.padding + 10e-6)
        self.triplane_coords=triplane_coords.float().cuda()

    def forward(self,image_feat,proj_mat):
        image_feat=self.img_proj(image_feat)
        batch_size=image_feat.shape[0]
        triplane_coords=self.triplane_coords.unsqueeze(0).expand(batch_size,-1,-1,-1) #B,192,64,3
        #print(torch.amin(triplane_coords),torch.amax(triplane_coords))
        coord_homo=torch.cat([triplane_coords,torch.ones((batch_size,triplane_coords.shape[1],triplane_coords.shape[2],1)).float().cuda()],dim=-1)
        coord_inimg=torch.einsum('bhwc,bck->bhwk',coord_homo,proj_mat.transpose(1,2))
        x=coord_inimg[:,:,:,0]/coord_inimg[:,:,:,2]
        y=coord_inimg[:,:,:,1]/coord_inimg[:,:,:,2]
        x=(x/(224.0-1.0)-0.5)*2 #-1~1
        y=(y/(224.0-1.0)-0.5)*2 #-1~1
        dist=coord_inimg[:,:,:,2]

        xy=torch.cat([x[:,:,:,None],y[:,:,:,None]],dim=-1)
        #print(image_feat.shape,xy.shape)
        sample_feat=torch.nn.functional.grid_sample(image_feat,xy,align_corners=True,mode='bilinear')
        return sample_feat

def position_encoding(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1) #length,1
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))) #d_model//2, this is the frequency
    pe[:, 0::2] = torch.sin(position.float() * div_term) #length*(d_model//2)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class Image_Vox_Local_Sampler(nn.Module):
    def __init__(self,reso,padding=0.1,in_channels=1280,inner_channel=128,out_channels=64,n_heads=8):
        super().__init__()
        self.triplane_reso=reso
        self.padding=padding
        self.get_vox_coord()
        self.out_channels=out_channels
        self.img_proj=nn.Conv2d(in_channels=in_channels,out_channels=inner_channel,kernel_size=1)

        self.vox_process=nn.Sequential(
            nn.Conv3d(in_channels=inner_channel,out_channels=inner_channel,kernel_size=3,padding=1,),
        )
        self.k=nn.Linear(in_features=inner_channel,out_features=inner_channel)
        self.q=nn.Linear(in_features=inner_channel,out_features=inner_channel)
        self.v=nn.Linear(in_features=inner_channel,out_features=inner_channel)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)

        self.proj_out=nn.Conv2d(in_channels=inner_channel,out_channels=out_channels,kernel_size=1)
        self.condition_pe = position_encoding(inner_channel, self.triplane_reso).unsqueeze(0)
    def get_vox_coord(self):
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)

        X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
        vox_coor=torch.cat([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],dim=-1)
        vox_coor=vox_coor/(self.triplane_reso-1)
        vox_coor=(vox_coor-0.5)*2*(1+self.padding+10e-6)
        self.vox_coor=vox_coor.view(-1,3).float().cuda()


    def forward(self,triplane_feat,image_feat,proj_mat):
        xz_feat,xy_feat,yz_feat=torch.split(triplane_feat,triplane_feat.shape[2]//3,dim=2) #B,C,64,64
        image_feat=self.img_proj(image_feat)
        batch_size=image_feat.shape[0]
        vox_coords=self.vox_coor.unsqueeze(0).expand(batch_size,-1,-1) #B,64*64*64,3
        vox_homo=torch.cat([vox_coords,torch.ones((batch_size,self.triplane_reso**3,1)).float().cuda()],dim=-1)
        coord_inimg=torch.einsum('bhc,bck->bhk',vox_homo,proj_mat.transpose(1,2))
        x=coord_inimg[:,:,0]/coord_inimg[:,:,2]
        y=coord_inimg[:,:,1]/coord_inimg[:,:,2]
        x=(x/(224.0-1.0)-0.5)*2 #-1~1
        y=(y/(224.0-1.0)-0.5)*2 #-1~1

        xy=torch.cat([x[:,:,None],y[:,:,None]],dim=-1).unsqueeze(1).contiguous() #B, 1,64**3,2
        #print(image_feat.shape,xy.shape)
        grid_feat=torch.nn.functional.grid_sample(image_feat,xy,align_corners=True,mode='bilinear').squeeze(2).\
            view(batch_size,-1,self.triplane_reso,self.triplane_reso,self.triplane_reso) #B,C,1,64**3

        grid_feat=self.vox_process(grid_feat)
        xzy_grid=grid_feat.permute(0,4,2,3,1)
        xz_as_query=xz_feat.permute(0,2,3,1).reshape(batch_size*self.triplane_reso**2,1,-1)
        xz_as_key=xzy_grid.reshape(batch_size*self.triplane_reso**2,self.triplane_reso,-1)

        xyz_grid=grid_feat.permute(0,3,2,4,1)
        xy_as_query=xy_feat.permute(0,2,3,1).reshape(batch_size*self.triplane_reso**2,1,-1)
        xy_as_key = xyz_grid.reshape(batch_size * self.triplane_reso ** 2, self.triplane_reso, -1)

        yzx_grid = grid_feat.permute(0, 4, 3, 2, 1)
        yz_as_query = yz_feat.permute(0,2,3,1).reshape(batch_size*self.triplane_reso**2,1,-1)
        yz_as_key = yzx_grid.reshape(batch_size * self.triplane_reso ** 2, self.triplane_reso, -1)

        query=self.q(torch.cat([xz_as_query,xy_as_query,yz_as_query],dim=0))
        key=self.k(torch.cat([xz_as_key,xy_as_key,yz_as_key],dim=0))+self.condition_pe.to(xz_as_key.device)
        value=self.v(torch.cat([xz_as_key,xy_as_key,yz_as_key],dim=0))+self.condition_pe.to(xz_as_key.device)

        attn,_=self.attn(query,key,value)
        xz_plane,xy_plane,yz_plane=torch.split(attn,dim=0,split_size_or_sections=batch_size*self.triplane_reso**2)
        xz_plane=xz_plane.reshape(batch_size,self.triplane_reso,self.triplane_reso,-1).permute(0,3,1,2)
        xy_plane = xy_plane.reshape(batch_size, self.triplane_reso, self.triplane_reso, -1).permute(0, 3, 1, 2)
        yz_plane = yz_plane.reshape(batch_size, self.triplane_reso, self.triplane_reso, -1).permute(0, 3, 1, 2)

        triplane_wImg=torch.cat([xz_plane,xy_plane,yz_plane],dim=2)
        triplane_wImg=self.proj_out(triplane_wImg)
        #print(triplane_wImg.shape)

        return triplane_wImg

class Image_Direct_AttenwMask_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,
                 img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8):
        super().__init__()
        self.triplane_reso=reso
        self.vit_reso=vit_reso
        self.padding=padding
        self.n_heads=n_heads
        self.get_plane_expand_coord()
        self.get_vit_coords()
        self.out_channels=out_channels
        self.kernel_func=mask_kernel
        self.k=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.q=nn.Linear(in_features=triplane_in_channels,out_features=inner_channel)
        self.v=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)

        self.proj_out=nn.Linear(in_features=inner_channel,out_features=out_channels)
        self.image_pe = position_encoding(inner_channel, self.vit_reso**2+1).unsqueeze(0).cuda().float() #1,n_img*reso*reso,channel
        self.triplane_pe = position_encoding(inner_channel, 3*self.triplane_reso**2).unsqueeze(0).cuda().float()
    def get_plane_expand_coord(self):
        x = torch.arange(self.triplane_reso)/(self.triplane_reso-1)
        y = torch.arange(self.triplane_reso)/(self.triplane_reso-1)
        z = torch.arange(self.triplane_reso)/(self.triplane_reso-1)

        first,second,third=torch.meshgrid(x,y,z,indexing='xy')
        xyz_coords=torch.stack([first,second,third],dim=-1)#reso,reso,reso,3
        xyz_coords=(xyz_coords-0.5)*2*(1+self.padding+10e-6) #ordering yxz ->xyz
        xzy_coords=xyz_coords.clone().permute(2,1,0,3) #ordering zxy ->xzy
        yzx_coords=xyz_coords.clone().permute(2,0,1,3) #ordering zyx ->yzx

        # print(xyz_coords[0,0,0],xyz_coords[0,0,1],xyz_coords[1,0,0],xyz_coords[0,1,0])
        # print(xzy_coords[0, 0, 0], xzy_coords[0, 0, 1], xzy_coords[1, 0, 0], xzy_coords[0, 1, 0])
        # print(yzx_coords[0, 0, 0], yzx_coords[0, 0, 1], yzx_coords[1, 0, 0], yzx_coords[0, 1, 0])

        xyz_coords=xyz_coords.reshape(self.triplane_reso**3,-1)
        xzy_coords=xzy_coords.reshape(self.triplane_reso**3,-1)
        yzx_coords=yzx_coords.reshape(self.triplane_reso**3,-1)

        coords=torch.cat([xzy_coords,xyz_coords,yzx_coords],dim=0)
        self.plane_coords=coords.cuda().float()
        # self.xzy_coords=xzy_coords.cuda().float() #reso**3，3
        # self.xyz_coords=xyz_coords.cuda().float() #reso**3，3
        # self.yzx_coords=yzx_coords.cuda().float() #reso**3，3

    def get_vit_coords(self):
        x=torch.arange(self.vit_reso)
        y=torch.arange(self.vit_reso)

        X,Y=torch.meshgrid(x,y,indexing='xy')
        vit_coords=torch.stack([X,Y],dim=-1)
        self.vit_coords=vit_coords.view(self.vit_reso**2,2).cuda().float()

    def get_attn_mask(self,coords_proj,vit_coords,kernel_size=1.0):
        '''
        :param coords_proj: B,reso**3,2, in range of 0~1
        :param vit_coords: B,vit_reso**2,2, in range of 0~vit_reso
        :param kernel_size: 0.5, so that only one pixel will be select
        :return:
        '''
        bs=coords_proj.shape[0]
        coords_proj=coords_proj*(self.vit_reso-1)
        #print(torch.amin(coords_proj[0,0:self.triplane_reso**3]),torch.amax(coords_proj[0,0:self.triplane_reso**3]))
        dist=torch.cdist(coords_proj.float(),vit_coords.float())
        mask=self.kernel_func(dist,sigma=kernel_size).float() #True if valid, B,3*reso**3,vit_reso**2
        mask=mask.reshape(bs,3*self.triplane_reso**2,self.triplane_reso,self.vit_reso**2)
        mask=torch.sum(mask,dim=2)
        attn_mask=(mask==0)
        return attn_mask

    def forward(self,triplane_feat,image_feat,proj_mat):
        #xz_feat,xy_feat,yz_feat=torch.split(triplane_feat,triplane_feat.shape[2]//3,dim=2) #B,C,64,64
        batch_size=image_feat.shape[0]
        #print(self.plane_coords.shape)
        coords=self.plane_coords.unsqueeze(0).expand(batch_size,-1,-1)

        coords_homo=torch.cat([coords,torch.ones(batch_size,self.triplane_reso**3*3,1).float().cuda()],dim=-1)
        coords_inimg=torch.einsum('bhc,bck->bhk',coords_homo,proj_mat.transpose(1,2))
        coords_x=coords_inimg[:,:,0]/coords_inimg[:,:,2]/(224.0-1) #0~1
        coords_y=coords_inimg[:,:,1]/coords_inimg[:,:,2]/(224.0-1) #0~1
        coords_x=torch.clamp(coords_x,min=0.0,max=1.0)
        coords_y=torch.clamp(coords_y,min=0.0,max=1.0)
        #print(torch.amin(coords_x),torch.amax(coords_x))
        coords_proj=torch.stack([coords_x,coords_y],dim=-1)
        vit_coords=self.vit_coords.unsqueeze(0).expand(batch_size,-1,-1)
        attn_mask=torch.repeat_interleave(
            self.get_attn_mask(coords_proj,vit_coords,kernel_size=1.0),self.n_heads, 0
        )
        attn_mask = torch.cat([torch.zeros([attn_mask.shape[0], attn_mask.shape[1], 1]).cuda().bool(), attn_mask],
                              dim=-1)  # add global token
        #print(attn_mask.shape,torch.sum(attn_mask.float()))
        triplane_feat=triplane_feat.permute(0,2,3,1).view(batch_size,3*self.triplane_reso**2,-1)
        #print(triplane_feat.shape,self.triplane_pe.shape)
        query=self.q(triplane_feat)+self.triplane_pe
        key=self.k(image_feat)+self.image_pe
        value=self.v(image_feat)+self.image_pe
        #print(query.shape,key.shape,value.shape)
        attn,_=self.attn(query,key,value,attn_mask=attn_mask)
        #print(attn.shape)
        output=self.proj_out(attn).transpose(1,2).reshape(batch_size,-1,3*self.triplane_reso,self.triplane_reso)

        return output

class MultiImage_Direct_AttenwMask_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,
                 img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8,max_nimg=5):
        super().__init__()
        self.triplane_reso=reso
        self.vit_reso=vit_reso
        self.padding=padding
        self.n_heads=n_heads
        self.get_plane_expand_coord()
        self.get_vit_coords()
        self.out_channels=out_channels
        self.kernel_func=mask_kernel
        self.k=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.q=nn.Linear(in_features=triplane_in_channels,out_features=inner_channel)
        self.v=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)

        self.proj_out=nn.Linear(in_features=inner_channel,out_features=out_channels)
        self.image_pe = position_encoding(inner_channel, max_nimg*(self.vit_reso**2+1)).unsqueeze(0).cuda().float()
        self.triplane_pe = position_encoding(inner_channel, 3*self.triplane_reso**2).unsqueeze(0).cuda().float()
    def get_plane_expand_coord(self):
        x = torch.arange(self.triplane_reso)/(self.triplane_reso-1)
        y = torch.arange(self.triplane_reso)/(self.triplane_reso-1)
        z = torch.arange(self.triplane_reso)/(self.triplane_reso-1)

        first,second,third=torch.meshgrid(x,y,z,indexing='xy')
        xyz_coords=torch.stack([first,second,third],dim=-1)#reso,reso,reso,3
        xyz_coords=(xyz_coords-0.5)*2*(1+self.padding+10e-6) #ordering yxz ->xyz
        xzy_coords=xyz_coords.clone().permute(2,1,0,3) #ordering zxy ->xzy
        yzx_coords=xyz_coords.clone().permute(2,0,1,3) #ordering zyx ->yzx

        xyz_coords=xyz_coords.reshape(self.triplane_reso**3,-1)
        xzy_coords=xzy_coords.reshape(self.triplane_reso**3,-1)
        yzx_coords=yzx_coords.reshape(self.triplane_reso**3,-1)

        coords=torch.cat([xzy_coords,xyz_coords,yzx_coords],dim=0)
        self.plane_coords=coords.cuda().float()
        # self.xzy_coords=xzy_coords.cuda().float() #reso**3，3
        # self.xyz_coords=xyz_coords.cuda().float() #reso**3，3
        # self.yzx_coords=yzx_coords.cuda().float() #reso**3，3

    def get_vit_coords(self):
        x=torch.arange(self.vit_reso)
        y=torch.arange(self.vit_reso)

        X,Y=torch.meshgrid(x,y,indexing='xy')
        vit_coords=torch.stack([X,Y],dim=-1)
        self.vit_coords=vit_coords.view(self.vit_reso**2,2).cuda().float()

    def get_attn_mask(self,coords_proj,vit_coords,valid_frames,kernel_size=1.0):
        '''
        :param coords_proj: B,n_img,3*reso**3,2, in range of 0~vit_reso
        :param vit_coords:  B,n_img,vit_reso**2,2, in range of 0~vit_reso
        :param kernel_size: 0.5, so that only one pixel will be select
        :return:
        '''
        bs,n_img=coords_proj.shape[0],coords_proj.shape[1]
        coords_proj_flat=coords_proj.reshape(bs*n_img,3*self.triplane_reso**3,2)
        vit_coords_flat=vit_coords.reshape(bs*n_img,self.vit_reso**2,2)
        dist=torch.cdist(coords_proj_flat.float(),vit_coords_flat.float())
        mask=self.kernel_func(dist,sigma=kernel_size).float() #True if valid, B*n_img,3*reso**3,vit_reso**2
        mask=mask.reshape(bs,n_img,3*self.triplane_reso**2,self.triplane_reso,self.vit_reso**2)
        mask=torch.sum(mask,dim=3) #B,n_img,3*reso**2,vit_reso**2
        mask=torch.cat([torch.ones(size=mask.shape[0:3]).unsqueeze(3).float().cuda(),mask],dim=-1) #B,n_img,3*reso**2,vit_reso**2+1, add global mask
        mask[valid_frames == 0, :, :] = False
        mask=mask.permute(0,2,1,3).reshape(bs,3*self.triplane_reso**2,-1) #B,3*reso**2,n_img*(vit_resso**2+1)
        attn_mask=(mask==0) #invert the mask, False indicates valid, True indicates invalid
        return attn_mask

    def forward(self,triplane_feat,image_feat,proj_mat,valid_frames):
        '''image feat is bs,n_img,length,channel'''
        batch_size,n_img=image_feat.shape[0],image_feat.shape[1]
        img_length=image_feat.shape[2]
        image_feat_flat=image_feat.view(batch_size,n_img*img_length,-1)
        coords=self.plane_coords.unsqueeze(0).unsqueeze(1).expand(batch_size,n_img,-1,-1)

        coord_homo=torch.cat([coords,torch.ones(batch_size,n_img,self.triplane_reso**3*3,1).float().cuda()],dim=-1)
        #print(coord_homo.shape,proj_mat.shape)
        coord_inimg = torch.einsum('bjnc,bjck->bjnk', coord_homo, proj_mat.transpose(2, 3))
        x = coord_inimg[:, :, :, 0] / coord_inimg[:, :, :, 2]
        y = coord_inimg[:, :, :, 1] / coord_inimg[:, :, :, 2]
        x = x/(224.0-1)
        y = y/(224.0-1)
        coords_x=torch.clamp(x,min=0.0,max=1.0)*(self.vit_reso-1)
        coords_y=torch.clamp(y,min=0.0,max=1.0)*(self.vit_reso-1)
        coords_proj=torch.stack([coords_x,coords_y],dim=-1)
        vit_coords=self.vit_coords.unsqueeze(0).unsqueeze(1).expand(batch_size,n_img,-1,-1)
        attn_mask=torch.repeat_interleave(
            self.get_attn_mask(coords_proj,vit_coords,valid_frames,kernel_size=1.0),self.n_heads, 0
        )
        triplane_feat=triplane_feat.permute(0,2,3,1).view(batch_size,3*self.triplane_reso**2,-1)
        query=self.q(triplane_feat)+self.triplane_pe
        key=self.k(image_feat_flat)+self.image_pe
        value=self.v(image_feat_flat)+self.image_pe
        attn,_=self.attn(query,key,value,attn_mask=attn_mask)
        output=self.proj_out(attn).transpose(1,2).reshape(batch_size,-1,3*self.triplane_reso,self.triplane_reso)

        return output

class MultiImage_Fuse_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,
                 img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8):
        super().__init__()
        self.triplane_reso=reso
        self.vit_reso=vit_reso
        self.inner_channel=inner_channel
        self.padding=padding
        self.n_heads=n_heads
        self.get_vox_coord()
        self.get_vit_coords()
        self.out_channels=out_channels
        self.kernel_func=mask_kernel
        self.image_unflatten=nn.Unflatten(2,(vit_reso,vit_reso))
        self.k=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.q=nn.Linear(in_features=triplane_in_channels*3,out_features=inner_channel)
        self.v=nn.Linear(in_features=img_in_channels,out_features=inner_channel)

        #self.cross_attn=CrossAttention(query_dim=inner_channel,heads=8,dim_head=inner_channel//8)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)
        self.proj_out=nn.Linear(in_features=inner_channel,out_features=out_channels)
        self.image_pe = position_encoding(inner_channel, self.vit_reso**2)[None,None,:,:].cuda().float() #1,1,length,channel
        #self.image_pe = self.image_pe.reshape(1,max_nimg,self.vit_reso,self.vit_reso,inner_channel)
        self.triplane_pe = position_encoding(inner_channel, self.triplane_reso ** 3).unsqueeze(0).cuda().float()

    def get_vit_coords(self):
        x = torch.arange(self.vit_reso)
        y = torch.arange(self.vit_reso)

        X, Y = torch.meshgrid(x, y, indexing='xy')
        vit_coords = torch.stack([X, Y], dim=-1)
        self.vit_coords = vit_coords.cuda().float() #reso,reso,2

    def get_vox_coord(self):
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)

        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        vox_coor = torch.cat([X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]], dim=-1)
        self.vox_index = vox_coor.view(-1, 3).long().cuda()

        vox_coor = self.vox_index.float() / (self.triplane_reso - 1)
        vox_coor = (vox_coor - 0.5) * 2 * (1 + self.padding + 10e-6)
        self.vox_coor = vox_coor.view(-1, 3).float().cuda()

    def get_attn_mask(self,valid_frames):
        '''
        :param valid_frames: of shape B,n_img
        '''
        #print(valid_frames)
        #bs,n_img=valid_frames.shape[0:2]
        attn_mask=(valid_frames.float()==0)
        #attn_mask=attn_mask.unsqueeze(1).unsqueeze(2).expand(-1,self.triplane_reso**3,-1,-1) #B,1,n_img
        #attn_mask=attn_mask.reshape(bs*self.triplane_reso**3,-1,n_img).bool()
        attn_mask=torch.repeat_interleave(attn_mask.unsqueeze(1),self.triplane_reso**3,0)
        # print(attn_mask[self.triplane_reso**3*1+10])
        # print(attn_mask[self.triplane_reso ** 3 * 2+10])
        # print(attn_mask[self.triplane_reso ** 3 * 3+10])
        return attn_mask

    def forward(self,triplane_feat,image_feat,proj_mat,valid_frames):
        '''image feat is bs,n_img,length,channel'''
        batch_size,n_img=image_feat.shape[0],image_feat.shape[1]
        image_feat=image_feat[:,:,1:,:] #discard global feature

        #image_feat=image_feat.permute(0,1,3,4,2) #B,n_img,h,w,c
        image_k=self.k(image_feat)+self.image_pe #B,n_img,h,w,c
        image_v=self.v(image_feat)+self.image_pe #B,n_img,h,w,c
        image_k_v=torch.cat([image_k,image_v],dim=-1) #B,n_img,h,w,c
        unflat_k_v=self.image_unflatten(image_k_v).permute(0,4,1,2,3) #Bs,channel,n_img,reso,reso
        #unflat_k_v=image_k_v.permute(0,4,1,2,3)
        #vit_coords=self.vit_coords[None,None].expand(batch_size,n_img,-1,-1,-1) #Bs,n_img,reso,reso,2

        coords=self.vox_coor.unsqueeze(0).unsqueeze(1).expand(batch_size,n_img,-1,-1)
        coord_homo=torch.cat([coords,torch.ones(batch_size,n_img,self.triplane_reso**3,1).float().cuda()],dim=-1)
        coord_inimg = torch.einsum('bjnc,bjck->bjnk', coord_homo, proj_mat.transpose(2, 3))
        x = coord_inimg[:, :, :, 0] / coord_inimg[:, :, :, 2]
        y = coord_inimg[:, :, :, 1] / coord_inimg[:, :, :, 2]
        x = x/(224.0-1) #0~1
        y = y/(224.0-1)
        coords_proj=torch.stack([x,y],dim=-1)
        coords_proj=(coords_proj-0.5)*2
        img_index=((torch.arange(n_img)[None,:,None,None].expand(
            batch_size,-1,self.triplane_reso**3,-1).float().cuda()/(n_img-1))-0.5)*2 #Bs,n_img,64**3,1

        # img_index_feat=torch.arange(n_img)[None,:,None,None,None].expand(
        #     batch_size,-1,self.vit_reso,self.vit_reso,-1).float().cuda() #Bs,n_img,reso,reso,1
        #coords_feat=torch.cat([vit_coords,img_index_feat],dim=-1).permute(0,4,1,2,3)#Bs,n_img,reso,reso,3
        grid=torch.cat([coords_proj,img_index],dim=-1) #x,y,index
        grid=torch.clamp(grid,min=-1.0,max=1.0)
        sample_k_v = torch.nn.functional.grid_sample(unflat_k_v, grid.unsqueeze(1), align_corners=True, mode='bilinear').squeeze(2) #B,C,n_img,64**3
        xz_feat, xy_feat, yz_feat = torch.split(triplane_feat, split_size_or_sections=triplane_feat.shape[2] // 3,
                                                dim=2)  # B,C,64,64
        xz_vox_feat=xz_feat.unsqueeze(4).expand(-1,-1,-1,-1,self.triplane_reso)#.permute(0,1,3,4,2).reshape(batch_size,-1,self.triplane_reso**3).transpose(1,2) #zxy
        xz_vox_feat=rearrange(xz_vox_feat, 'b c z x y -> b (x y z) c')
        xy_vox_feat=xy_feat.unsqueeze(4).expand(-1,-1,-1,-1,self.triplane_reso)#.permute(0,1,3,2,4).reshape(batch_size,-1,self.triplane_reso**3).transpose(1,2) #yxz
        xy_vox_feat=rearrange(xy_vox_feat, 'b c y x z -> b (x y z) c')
        yz_vox_feat=yz_feat.unsqueeze(4).expand(-1,-1,-1,-1,self.triplane_reso)#.permute(0,1,4,3,2).reshape(batch_size,-1,self.triplane_reso**3).transpose(1,2) #zyx
        yz_vox_feat=rearrange(yz_vox_feat, 'b c z y x -> b (x y z) c')
        #xz_vox_feat = xz_feat[:, :, vox_index[:, 2], vox_index[:, 0]].transpose(1, 2)  # B,C,64*64*64
        #xy_vox_feat = xy_feat[:, :, vox_index[:, 1], vox_index[:, 0]].transpose(1, 2)
        #yz_vox_feat = yz_feat[:, :, vox_index[:, 2], vox_index[:, 1]].transpose(1, 2)

        triplane_expand_feat = torch.cat([xz_vox_feat, xy_vox_feat, yz_vox_feat], dim=-1)  # B,64*64*64,3*C
        triplane_query = self.q(triplane_expand_feat) + self.triplane_pe
        k_v=rearrange(sample_k_v, 'b c n k -> (b k) n c')
        #k_v=sample_k_v.permute(0,3,2,1).reshape(batch_size*self.triplane_reso**3,n_img,-1) #B*64**3,n_img,C
        k=k_v[:,:,0:self.inner_channel]
        v=k_v[:,:,self.inner_channel:]
        q=rearrange(triplane_query,'b k c -> (b k) 1 c')
        #q=triplane_query.view(batch_size*self.triplane_reso**3,1,-1)
        #k,v is of shape, B*reso**3,k,channel, q is of shape B*reso**3,1,channel
        #attn mask should be B*reso**3*n_heads,1,k
        #attn_mask=torch.repeat_interleave(self.get_attn_mask(valid_frames),self.n_heads,0)
        #print(q.shape,k.shape,v.shape)
        attn_out,_=self.cross_attn(q,k,v)#attn_mask=attn_mask) #fuse multi-view feature
        #volume=attn_out.view(batch_size,self.triplane_reso,self.triplane_reso,self.triplane_reso,-1) #B,reso,reso,reso,channel
        #print(attn_out.shape)
        volume=rearrange(attn_out,'(b x y z) 1 c -> b x y z c',x=self.triplane_reso,y=self.triplane_reso,z=self.triplane_reso)
        #xz_feat = torch.mean(volume, dim=2).transpose(1,2) #B,reso,reso,C
        xz_feat = reduce(volume, "b x y z c -> b z x c", 'mean')
        #xy_feat = torch.mean(volume, dim=3).transpose(1,2) #B,reso,reso,C
        xy_feat= reduce(volume, 'b x y z c -> b y x c', 'mean')
        #yz_feat = torch.mean(volume, dim=1).transpose(1,2) #B,reso,reso,C
        yz_feat=reduce(volume, 'b x y z c -> b z y c', 'mean')
        triplane_out = torch.cat([xz_feat, xy_feat, yz_feat], dim=1) #B,reso*3,reso,C
        #print(triplane_out.shape)
        triplane_out = self.proj_out(triplane_out)
        triplane_out = triplane_out.permute(0,3,1,2)
        #print(triplane_out.shape)
        return triplane_out

class MultiImage_TriFuse_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,
                 img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8,max_nimg=5):
        super().__init__()
        self.triplane_reso=reso
        self.vit_reso=vit_reso
        self.inner_channel=inner_channel
        self.padding=padding
        self.n_heads=n_heads
        self.get_triplane_coord()
        self.get_vit_coords()
        self.out_channels=out_channels
        self.kernel_func=mask_kernel
        self.image_unflatten=nn.Unflatten(2,(vit_reso,vit_reso))
        self.k=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.q=nn.Linear(in_features=triplane_in_channels,out_features=inner_channel)
        self.v=nn.Linear(in_features=img_in_channels,out_features=inner_channel)

        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)
        self.proj_out=nn.Conv2d(in_channels=inner_channel,out_channels=out_channels,kernel_size=1)
        self.image_pe = position_encoding(inner_channel, self.vit_reso**2)[None,None,:,:].expand(-1,max_nimg,-1,-1).cuda().float() #B,n_img,length,channel
        self.triplane_pe = position_encoding(inner_channel, self.triplane_reso ** 2*3).unsqueeze(0).cuda().float()

    def get_vit_coords(self):
        x = torch.arange(self.vit_reso)
        y = torch.arange(self.vit_reso)

        X, Y = torch.meshgrid(x, y, indexing='xy')
        vit_coords = torch.stack([X, Y], dim=-1)
        self.vit_coords = vit_coords.cuda().float() #reso,reso,2

    def get_triplane_coord(self):
        '''xz plane firstly, z is at the '''
        x = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)
        X, Z = torch.meshgrid(x, z, indexing='xy')
        xz_coords = torch.cat(
            [X[:, :, None], torch.ones_like(X[:, :, None]) * (self.triplane_reso - 1) / 2, Z[:, :, None]],
            dim=-1)  # in xyz order

        '''xy plane'''
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        xy_coords = torch.cat(
            [X[:, :, None], Y[:, :, None], torch.ones_like(X[:, :, None]) * (self.triplane_reso - 1) / 2],
            dim=-1)  # in xyz order

        '''yz plane'''
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)
        Y, Z = torch.meshgrid(y, z, indexing='xy')
        yz_coords = torch.cat(
            [torch.ones_like(Y[:, :, None]) * (self.triplane_reso - 1) / 2, Y[:, :, None], Z[:, :, None]], dim=-1)

        triplane_coords = torch.cat([xz_coords, xy_coords, yz_coords], dim=0)
        triplane_coords = triplane_coords / (self.triplane_reso - 1)
        triplane_coords = (triplane_coords - 0.5) * 2 * (1 + self.padding + 10e-6)
        self.triplane_coords = triplane_coords.view(-1,3).float().cuda()
    def forward(self,triplane_feat,image_feat,proj_mat,valid_frames):
        '''image feat is bs,n_img,length,channel'''
        batch_size,n_img=image_feat.shape[0],image_feat.shape[1]
        image_feat=image_feat[:,:,1:,:] #discard global feature
        #print(image_feat.shape)

        #image_feat=image_feat.permute(0,1,3,4,2) #B,n_img,h,w,c
        image_k=self.k(image_feat)+self.image_pe #B,n_img,h,w,c
        image_v=self.v(image_feat)+self.image_pe #B,n_img,h,w,c
        image_k_v=torch.cat([image_k,image_v],dim=-1) #B,n_img,h,w,c
        unflat_k_v=self.image_unflatten(image_k_v).permute(0,4,1,2,3) #Bs,channel,n_img,reso,reso

        coords=self.triplane_coords.unsqueeze(0).unsqueeze(1).expand(batch_size,n_img,-1,-1)
        coord_homo=torch.cat([coords,torch.ones(batch_size,n_img,self.triplane_reso**2*3,1).float().cuda()],dim=-1)
        coord_inimg = torch.einsum('bjnc,bjck->bjnk', coord_homo, proj_mat.transpose(2, 3))
        x = coord_inimg[:, :, :, 0] / coord_inimg[:, :, :, 2]
        y = coord_inimg[:, :, :, 1] / coord_inimg[:, :, :, 2]
        x = x/(224.0-1) #0~1
        y = y/(224.0-1)
        coords_proj=torch.stack([x,y],dim=-1)
        coords_proj=(coords_proj-0.5)*2
        img_index=((torch.arange(n_img)[None,:,None,None].expand(
            batch_size,-1,self.triplane_reso**2*3,-1).float().cuda()/(n_img-1))-0.5)*2 #Bs,n_img,64**3,1

        grid=torch.cat([coords_proj,img_index],dim=-1) #x,y,index
        grid=torch.clamp(grid,min=-1.0,max=1.0)
        sample_k_v = torch.nn.functional.grid_sample(unflat_k_v, grid.unsqueeze(1), align_corners=True, mode='bilinear').squeeze(2) #B,C,n_img,64**3

        triplane_flat_feat=rearrange(triplane_feat,'b c h w -> b (h w) c')
        triplane_query = self.q(triplane_flat_feat) + self.triplane_pe

        k_v=rearrange(sample_k_v, 'b c n k -> (b k) n c')
        k=k_v[:,:,0:self.inner_channel]
        v=k_v[:,:,self.inner_channel:]
        q=rearrange(triplane_query,'b k c -> (b k) 1 c')
        attn_out,_=self.cross_attn(q,k,v)
        triplane_out=rearrange(attn_out,'(b h w) 1 c -> b c h w',b=batch_size,h=self.triplane_reso*3,w=self.triplane_reso)
        triplane_out = self.proj_out(triplane_out)
        return triplane_out


class MultiImage_Global_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,
                 img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8,max_nimg=5):
        super().__init__()
        self.triplane_reso=reso
        self.vit_reso=vit_reso
        self.inner_channel=inner_channel
        self.padding=padding
        self.n_heads=n_heads
        self.out_channels=out_channels
        self.k=nn.Linear(in_features=img_in_channels,out_features=inner_channel)
        self.q=nn.Linear(in_features=triplane_in_channels,out_features=inner_channel)
        self.v=nn.Linear(in_features=img_in_channels,out_features=inner_channel)

        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)
        self.proj_out=nn.Linear(in_features=inner_channel,out_features=out_channels)
        self.image_pe = position_encoding(inner_channel, self.vit_reso**2)[None,None,:,:].expand(-1,max_nimg,-1,-1).cuda().float() #B,n_img,length,channel
        self.triplane_pe = position_encoding(inner_channel, self.triplane_reso**2*3).unsqueeze(0).cuda().float()
    def forward(self,triplane_feat,image_feat,proj_mat,valid_frames):
        '''image feat is bs,n_img,length,channel
            triplane feat is bs,C,H*3,W
        '''
        batch_size,n_img=image_feat.shape[0],image_feat.shape[1]
        L=image_feat.shape[2]-1
        image_feat=image_feat[:,:,1:,:] #discard global feature

        image_k=self.k(image_feat)+self.image_pe #B,n_img,h*w,c
        image_v=self.v(image_feat)+self.image_pe #B,n_img,h*w,c
        image_k=image_k.view(batch_size,n_img*L,-1)
        image_v=image_v.view(batch_size,n_img*L,-1)

        triplane_flat_feat=rearrange(triplane_feat,"b c h w -> b (h w) c")
        triplane_query = self.q(triplane_flat_feat) + self.triplane_pe
        #print(triplane_query.shape,image_k.shape,image_v.shape)
        attn_out,_=self.cross_attn(triplane_query,image_k,image_v)
        triplane_flat_out = self.proj_out(attn_out)
        triplane_out=rearrange(triplane_flat_out,"b (h w) c -> b c h w",h=self.triplane_reso*3,w=self.triplane_reso)

        return triplane_out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q,k,v):
        h = self.heads

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class Image_Vox_Local_Sampler_Pooling(nn.Module):
    def __init__(self,reso,padding=0.1,in_channels=1280,inner_channel=128,out_channels=64,stride=4):
        super().__init__()
        self.triplane_reso=reso
        self.padding=padding
        self.get_vox_coord()
        self.out_channels=out_channels
        self.img_proj=nn.Conv2d(in_channels=in_channels,out_channels=inner_channel,kernel_size=1)

        self.vox_process=nn.Sequential(
            nn.Conv3d(in_channels=inner_channel,out_channels=inner_channel,kernel_size=3,padding=1)
        )
        self.xz_conv=nn.Sequential(
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((1,stride,1),stride=(1,stride,1)), #8
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((1,stride,1), stride=(1,stride,1)), #2
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
        )
        self.xy_conv = nn.Sequential(
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((1, 1, stride), stride=(1, 1, stride)),  # 8
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((1, 1, stride), stride=(1, 1, stride)),  # 2
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
        )
        self.yz_conv = nn.Sequential(
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((stride, 1, 1), stride=(stride, 1, 1)),  # 8
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
            nn.AvgPool3d((stride, 1, 1), stride=(stride, 1, 1)),  # 2
            nn.BatchNorm3d(inner_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, padding=1),
        )
        self.roll_out_conv=RollOut_Conv(in_channels=inner_channel,out_channels=out_channels)
        #self.proj_out=nn.Conv2d(in_channels=inner_channel,out_channels=out_channels,kernel_size=1)
    def get_vox_coord(self):
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)

        X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
        vox_coor=torch.cat([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],dim=-1)
        vox_coor=vox_coor/(self.triplane_reso-1)
        vox_coor=(vox_coor-0.5)*2*(1+self.padding+10e-6)
        self.vox_coor=vox_coor.view(-1,3).float().cuda()


    def forward(self,image_feat,proj_mat):
        image_feat=self.img_proj(image_feat)
        batch_size=image_feat.shape[0]
        vox_coords=self.vox_coor.unsqueeze(0).expand(batch_size,-1,-1) #B,64*64*64,3
        vox_homo=torch.cat([vox_coords,torch.ones((batch_size,self.triplane_reso**3,1)).float().cuda()],dim=-1)
        coord_inimg=torch.einsum('bhc,bck->bhk',vox_homo,proj_mat.transpose(1,2))
        x=coord_inimg[:,:,0]/coord_inimg[:,:,2]
        y=coord_inimg[:,:,1]/coord_inimg[:,:,2]
        x=(x/(224.0-1.0)-0.5)*2 #-1~1
        y=(y/(224.0-1.0)-0.5)*2 #-1~1

        xy=torch.cat([x[:,:,None],y[:,:,None]],dim=-1).unsqueeze(1).contiguous() #B, 1,64**3,2
        #print(image_feat.shape,xy.shape)
        grid_feat=torch.nn.functional.grid_sample(image_feat,xy,align_corners=True,mode='bilinear').squeeze(2).\
            view(batch_size,-1,self.triplane_reso,self.triplane_reso,self.triplane_reso) #B,C,1,64**3

        grid_feat=self.vox_process(grid_feat)
        xz_feat=torch.mean(self.xz_conv(grid_feat),dim=3).transpose(2,3)
        xy_feat=torch.mean(self.xy_conv(grid_feat),dim=4).transpose(2,3)
        yz_feat=torch.mean(self.yz_conv(grid_feat),dim=2).transpose(2,3)
        triplane_wImg=torch.cat([xz_feat,xy_feat,yz_feat],dim=2)
        #print(triplane_wImg.shape)

        return self.roll_out_conv(triplane_wImg)

class Image_ExpandVox_attn_Sampler(nn.Module):
    def __init__(self,reso,vit_reso=16,padding=0.1,triplane_in_channels=64,img_in_channels=1280,inner_channel=128,out_channels=64,n_heads=8):
        super().__init__()
        self.triplane_reso=reso
        self.padding=padding
        self.vit_reso=vit_reso
        self.get_vox_coord()
        self.get_vit_coords()
        self.out_channels=out_channels
        self.n_heads=n_heads

        self.kernel_func = mask_kernel_close_false
        self.k = nn.Linear(in_features=img_in_channels, out_features=inner_channel)
        # self.q_xz = nn.Conv2d(in_channels=triplane_in_channels,out_channels=inner_channel,kernel_size=1)
        # self.q_xy = nn.Conv2d(in_channels=triplane_in_channels,out_channels=inner_channel,kernel_size=1)
        # self.q_yz = nn.Conv2d(in_channels=triplane_in_channels,out_channels=inner_channel,kernel_size=1)
        self.q=nn.Linear(in_features=triplane_in_channels*3,out_features=inner_channel)

        self.v = nn.Linear(in_features=img_in_channels, out_features=inner_channel)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channel, num_heads=n_heads, batch_first=True)
        self.out_proj=nn.Linear(in_features=inner_channel,out_features=out_channels)

        self.triplane_pe = position_encoding(inner_channel, self.triplane_reso ** 3).unsqueeze(0).cuda().float()
        self.image_pe = position_encoding(inner_channel, self.vit_reso ** 2+1).unsqueeze(0).cuda().float()
    def get_vox_coord(self):
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)

        X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
        vox_coor=torch.cat([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],dim=-1)
        self.vox_index=vox_coor.view(-1,3).long().cuda()


        vox_coor = self.vox_index.float() / (self.triplane_reso - 1)
        vox_coor = (vox_coor - 0.5) * 2 * (1 + self.padding + 10e-6)
        self.vox_coor = vox_coor.view(-1, 3).float().cuda()
        # print(self.vox_coor[0])
        # print(self.vox_coor[self.triplane_reso**2])#x should increase
        # print(self.vox_coor[self.triplane_reso]) #y should increase
        # print(self.vox_coor[1])#z should increase

    def get_vit_coords(self):
        x=torch.arange(self.vit_reso)
        y=torch.arange(self.vit_reso)

        X,Y=torch.meshgrid(x,y,indexing='xy')
        vit_coords=torch.stack([X,Y],dim=-1)
        self.vit_coords=vit_coords.view(self.vit_reso**2,2).cuda().float()

    def compute_attn_mask(self,proj_coords,vit_coords,kernel_size=1.0):
        dist = torch.cdist(proj_coords.float(), vit_coords.float())
        mask = self.kernel_func(dist, sigma=kernel_size)  # True if valid, B,reso**3,vit_reso**2
        return mask


    def forward(self,triplane_feat,image_feat,proj_mat):
        xz_feat, xy_feat, yz_feat = torch.split(triplane_feat, split_size_or_sections=triplane_feat.shape[2] // 3, dim=2)  # B,C,64,64
        #xz_feat=self.q_xz(xz_feat)
        #xy_feat=self.q_xy(xy_feat)
        #yz_feat=self.q_yz(yz_feat)
        batch_size=image_feat.shape[0]
        vox_index=self.vox_index #64*64*64,3
        xz_vox_feat=xz_feat[:,:,vox_index[:,2],vox_index[:,0]].transpose(1,2) #B,C,64*64*64
        xy_vox_feat=xy_feat[:,:,vox_index[:,1],vox_index[:,0]].transpose(1,2)
        yz_vox_feat=yz_feat[:,:,vox_index[:,2],vox_index[:,1]].transpose(1,2)
        triplane_expand_feat=torch.cat([xz_vox_feat,xy_vox_feat,yz_vox_feat],dim=-1)#B,C,64*64*64,3
        triplane_query=self.q(triplane_expand_feat)+self.triplane_pe

        '''compute projection'''
        vox_coords=self.vox_coor.unsqueeze(0).expand(batch_size,-1,-1) #
        vox_homo = torch.cat([vox_coords, torch.ones((batch_size, self.triplane_reso ** 3, 1)).float().cuda()], dim=-1)
        coord_inimg = torch.einsum('bhc,bck->bhk', vox_homo, proj_mat.transpose(1, 2))
        x = coord_inimg[:, :, 0] / coord_inimg[:, :, 2]
        y = coord_inimg[:, :, 1] / coord_inimg[:, :, 2]
        #
        x = x / (224.0 - 1.0)  * (self.vit_reso-1)  # 0~self.vit_reso-1
        y = y / (224.0 - 1.0)  * (self.vit_reso-1)  # 0~self.vit_reso-1 #B,N
        xy=torch.stack([x,y],dim=-1) #B,64*64*64,2
        xy=torch.clamp(xy,min=0,max=self.vit_reso-1)
        vit_coords=self.vit_coords.unsqueeze(0).expand(batch_size,-1,-1) #B, 16*16,2
        attn_mask=torch.repeat_interleave(self.compute_attn_mask(xy,vit_coords,kernel_size=0.5),
                                          self.n_heads,0) #B*n_heads, reso**3, vit_reso**2

        k=self.k(image_feat)+self.image_pe
        v=self.v(image_feat)+self.image_pe
        attn_mask=torch.cat([torch.zeros([attn_mask.shape[0],attn_mask.shape[1],1]).cuda().bool(),attn_mask],dim=-1) #add empty token to each key and value
        vox_feat,_=self.attn(triplane_query,k,v,attn_mask=attn_mask) #B,reso**3,C
        feat_volume=self.out_proj(vox_feat).transpose(1,2).reshape(batch_size,-1,self.triplane_reso,
                                                                   self.triplane_reso,self.triplane_reso)
        xz_feat=torch.mean(feat_volume,dim=3).transpose(2,3)
        xy_feat=torch.mean(feat_volume,dim=4).transpose(2,3)
        yz_feat=torch.mean(feat_volume,dim=2).transpose(2,3)
        triplane_out=torch.cat([xz_feat,xy_feat,yz_feat],dim=2)
        return triplane_out

class Multi_Image_Fusion(nn.Module):
    def __init__(self,reso,image_reso=16,padding=0.1,img_channels=1280,triplane_channel=64,inner_channels=128,output_channel=64,n_heads=8):
        super().__init__()
        self.triplane_reso=reso
        self.image_reso=image_reso
        self.padding=padding
        self.get_triplane_coord()
        self.get_vit_coords()
        self.img_proj=nn.Conv3d(in_channels=img_channels,out_channels=512,kernel_size=1)
        self.kernel_func=mask_kernel

        self.q = nn.Linear(in_features=triplane_channel, out_features=inner_channels, bias=False)
        self.k = nn.Linear(in_features=512, out_features=inner_channels)
        self.v = nn.Linear(in_features=512, out_features=inner_channels)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=inner_channels, num_heads=n_heads, batch_first=True)
        self.out_proj=nn.Linear(in_features=inner_channels,out_features=output_channel)
        self.n_heads=n_heads

    def get_triplane_coord(self):
        '''xz plane firstly, z is at the '''
        x=torch.arange(self.triplane_reso)
        z=torch.arange(self.triplane_reso)
        X,Z=torch.meshgrid(x,z,indexing='xy')
        xz_coords=torch.cat([X[:,:,None],torch.ones_like(X[:,:,None])*(self.triplane_reso-1)/2,Z[:,:,None]],dim=-1) #in xyz order

        '''xy plane'''
        x = torch.arange(self.triplane_reso)
        y = torch.arange(self.triplane_reso)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        xy_coords = torch.cat([X[:, :, None],  Y[:, :, None],torch.ones_like(X[:, :, None])*(self.triplane_reso-1)/2], dim=-1)  # in xyz order

        '''yz plane'''
        y = torch.arange(self.triplane_reso)
        z = torch.arange(self.triplane_reso)
        Y,Z = torch.meshgrid(y,z,indexing='xy')
        yz_coords= torch.cat([torch.ones_like(Y[:, :, None])*(self.triplane_reso-1)/2,Y[:,:,None],Z[:,:,None]], dim=-1)

        triplane_coords=torch.cat([xz_coords,xy_coords,yz_coords],dim=0)
        triplane_coords=triplane_coords/(self.triplane_reso-1)
        triplane_coords=(triplane_coords-0.5)*2*(1 + self.padding + 10e-6)
        self.triplane_coords=triplane_coords.float().cuda()

    def get_vit_coords(self):
        x=torch.arange(self.image_reso)
        y=torch.arange(self.image_reso)
        X,Y=torch.meshgrid(x,y,indexing='xy')
        vit_coords=torch.cat([X[:,:,None],Y[:,:,None]],dim=-1)
        self.vit_coords=vit_coords.float().cuda() #in x,y order

    def compute_attn_mask(self,proj_coord,vit_coords,valid_frames,kernel_size=2.0):
        '''
        :param proj_coord: B,K,H,W,2
        :param vit_coords: H,W,2
        :return:
        '''
        B,K=proj_coord.shape[0:2]
        vit_coords_expand=vit_coords[None,None,:,:,:].expand(B,K,-1,-1,-1)

        proj_coord=proj_coord.view(B*K,proj_coord.shape[2]*proj_coord.shape[3],proj_coord.shape[4])
        vit_coords_expand=vit_coords_expand.view(B*K,self.image_reso*self.image_reso,2)
        attn_mask=self.kernel_func(torch.cdist(proj_coord,vit_coords_expand),sigma=float(kernel_size))
        attn_mask=attn_mask.reshape(B,K,proj_coord.shape[1],vit_coords_expand.shape[1])
        valid_expand=valid_frames[:,:,None,None]
        attn_mask[valid_frames>0,:,:]=True
        attn_mask=attn_mask.permute(0,2,1,3)
        attn_mask=attn_mask.reshape(B,proj_coord.shape[1],K*vit_coords_expand.shape[1])
        atten_index=torch.where(attn_mask[0,0]==False)
        return attn_mask


    def forward(self,triplane_feat,image_feat,proj_mat,valid_frames):
        '''
        :param image_feat: B,C,K,16,16
        :param proj_mat: B,K,4,4
        :param valid_frames: B,K, true if have image, used to compute attn_mask for transformer
        :return:
        '''
        image_feat=self.img_proj(image_feat)
        batch_size=image_feat.shape[0] #K is number of frames
        K=image_feat.shape[2]
        triplane_coords=self.triplane_coords.unsqueeze(0).unsqueeze(1).expand(batch_size,K,-1,-1,-1) #B,K,192,64,3
        #print(torch.amin(triplane_coords),torch.amax(triplane_coords))
        coord_homo=torch.cat([triplane_coords,torch.ones((batch_size,K,triplane_coords.shape[2],triplane_coords.shape[3],1)).float().cuda()],dim=-1)
        #print(coord_homo.shape,proj_mat.shape)
        coord_inimg=torch.einsum('bjhwc,bjck->bjhwk',coord_homo,proj_mat.transpose(2,3))
        x=coord_inimg[:,:,:,:,0]/coord_inimg[:,:,:,:,2]
        y=coord_inimg[:,:,:,:,1]/coord_inimg[:,:,:,:,2]
        x=x/(224.0-1.0)*(self.image_reso-1)
        y=y/(224.0-1.0)*(self.image_reso-1)

        xy=torch.cat([x[...,None],y[...,None]],dim=-1) #B,K,H,W,2
        image_value=image_feat.view(image_feat.shape[0],image_feat.shape[1],-1).transpose(1,2)
        triplane_query=triplane_feat.view(triplane_feat.shape[0],triplane_feat.shape[1],-1).transpose(1,2)
        valid_frames=1.0-valid_frames.float()
        attn_mask=torch.repeat_interleave(self.compute_attn_mask(xy,self.vit_coords,valid_frames),
                                          self.n_heads,dim=0)

        q=self.q(triplane_query)
        k=self.k(image_value)
        v=self.v(image_value)
        #print(q.shape,k.shape,v.shape)

        attn,_=self.attn(q,k,v,attn_mask=attn_mask)
        #print(attn.shape)
        output=self.out_proj(attn).transpose(1,2).reshape(batch_size,-1,triplane_feat.shape[2],triplane_feat.shape[3])
        #print(output.shape)
        return output


if __name__=="__main__":
    # import sys
    # sys.path.append("../..")
    # from datasets.SingleView_dataset import Object_PartialPoints_Img
    # from datasets.transforms import Aug_with_Tran
    # #sampler=#Image_Vox_Local_Sampler_Pooling(reso=64,padding=0.1,out_channels=64,stride=4).cuda().float()
    # sampler=Image_ExpandVox_attn_Sampler(reso=32,vit_reso=16,padding=0.1,img_in_channels=1280,triplane_in_channels=64,inner_channel=64
    #                                         ,out_channels=64,n_heads=8).cuda().float()
    # # sampler=Image_Direct_AttenwMask_Sampler(reso=32,vit_reso=16,padding=0.1,img_in_channels=1280,triplane_in_channels=128,inner_channel=128
    # #                                          ,out_channels=64,n_heads=8).cuda().float()
    # dataset_config = {
    #     "data_path": "/data1/haolin/datasets",
    #     "surface_size": 20000,
    #     "par_pc_size": 4096,
    #     "load_proj_mat": True,
    # }
    # transform = Aug_with_Tran()
    # datasets = Object_PartialPoints_Img(dataset_config['data_path'], split_filename="val_par_img.json", split='val',
    #                                         transform=transform, sampling=False,
    #                                         num_samples=1024, return_surface=True, ret_sample=True,
    #                                         surface_sampling=True, par_pc_size=dataset_config['par_pc_size'],
    #                                         surface_size=dataset_config['surface_size'],
    #                                         load_proj_mat=dataset_config['load_proj_mat'], load_image=True,
    #                                         load_org_img=False, load_triplane=True, replica=1)
    #
    # dataloader = torch.utils.data.DataLoader(
    #     datasets=datasets,
    #     batch_size=64,
    #     shuffle=True
    # )
    # iterator = dataloader.__iter__()
    # data_batch = iterator.next()
    # unflatten = torch.nn.Unflatten(1, (16, 16))
    # image = data_batch['image'][:,:,:].cuda().float()
    # #image=unflatten(image).permute(0,3,1,2)
    # proj_mat = data_batch['proj_mat'].cuda().float()
    # triplane_feat=torch.randn((64,64,32*3,32)).cuda().float()
    # sampler(triplane_feat,image,proj_mat)
    # memory_usage=torch.cuda.max_memory_allocated() / MB
    # print("memory usage %f mb"%(memory_usage))


    import sys
    sys.path.append("../..")
    from datasets.SingleView_dataset import Object_PartialPoints_MultiImg
    from datasets.transforms import Aug_with_Tran

    dataset_config = {
        "data_path": "/data1/haolin/datasets",
        "surface_size": 20000,
        "par_pc_size": 4096,
        "load_proj_mat": True,
    }
    transform = Aug_with_Tran()
    dataset = Object_PartialPoints_MultiImg(dataset_config['data_path'], split_filename="train_par_img.json", split='train',
                                            transform=transform, sampling=False,
                                            num_samples=1024, return_surface=True, ret_sample=True,
                                            surface_sampling=True, par_pc_size=dataset_config['par_pc_size'],
                                            surface_size=dataset_config['surface_size'],
                                            load_proj_mat=dataset_config['load_proj_mat'], load_image=True,
                                            load_org_img=False, load_triplane=True, replica=1)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False
    )
    iterator = dataloader.__iter__()
    data_batch = iterator.next()
    #unflatten = torch.nn.Unflatten(2, (16, 16))
    image = data_batch['image'][:,:,:,:].cuda().float()
    #image=unflatten(image).permute(0,4,1,2,3)
    proj_mat = data_batch['proj_mat'].cuda().float()
    valid_frames = data_batch['valid_frames'].cuda().float()
    triplane_feat=torch.randn((10,128,32*3,32)).cuda().float()

    # fusion_module=MultiImage_Fuse_Sampler(reso=32,vit_reso=16,padding=0.1,img_in_channels=1280,triplane_in_channels=128,inner_channel=128
    #                                           ,out_channels=64,n_heads=8).cuda().float()
    fusion_module=MultiImage_Global_Sampler(reso=32,vit_reso=16,padding=0.1,img_in_channels=1280,triplane_in_channels=128,inner_channel=128
                                               ,out_channels=64,n_heads=8).cuda().float()
    fusion_module(triplane_feat,image,proj_mat,valid_frames)
    memory_usage=torch.cuda.max_memory_allocated() / MB
    print("memory usage %f mb"%(memory_usage))
