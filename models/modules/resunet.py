import torch
import torch.nn as nn
from .unet import RollOut_Conv
from .Positional_Embedding import PositionalEmbedding
import torch.nn.functional as F
from .utils import zero_module
from .image_sampler import MultiImage_Fuse_Sampler, MultiImage_Global_Sampler,MultiImage_TriFuse_Sampler

class ResidualConv_MultiImgAtten(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, reso=64,
                 vit_reso=16,t_input_dim=256,img_in_channels=1280,use_attn=True,triplane_padding=0.1,
                 norm="batch"):
        super(ResidualConv_MultiImgAtten, self).__init__()
        self.use_attn=use_attn

        if norm=="batch":
            norm_layer=nn.BatchNorm2d
        elif norm==None:
            norm_layer=nn.Identity

        self.conv_block = nn.Sequential(
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, padding=padding
            )
        )
        self.out_layer=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            norm_layer(output_dim),
        )
        self.roll_out_conv=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            RollOut_Conv(output_dim, output_dim),
        )
        if self.use_attn:
            self.img_sampler = MultiImage_Fuse_Sampler(inner_channel=output_dim, triplane_in_channels=output_dim,
                                                             img_in_channels=img_in_channels,reso=reso,vit_reso=vit_reso,
                                                             out_channels=output_dim,padding=triplane_padding)
        self.down_conv=nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding)

        self.map_layer0 = nn.Linear(in_features=t_input_dim, out_features=output_dim)
        self.map_layer1 = nn.Linear(in_features=output_dim, out_features=output_dim)
    def forward(self, x,t_emb,img_feat,proj_mat,valid_frames):
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        t_emb = t_emb[:,:,None,None]

        out=self.conv_block(x)+t_emb
        out=self.out_layer(out)
        feature=out+self.conv_skip(x)
        feature = self.roll_out_conv(feature)
        if self.use_attn:
            feature=self.img_sampler(feature,img_feat,proj_mat,valid_frames)+feature #skip connect
        feature=self.down_conv(feature)

        return feature

class ResidualConv_TriMultiImgAtten(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, reso=64,
                 vit_reso=16,t_input_dim=256,img_in_channels=1280,use_attn=True,triplane_padding=0.1,
                 norm="batch"):
        super(ResidualConv_TriMultiImgAtten, self).__init__()
        self.use_attn=use_attn

        if norm=="batch":
            norm_layer=nn.BatchNorm2d
        elif norm==None:
            norm_layer=nn.Identity

        self.conv_block = nn.Sequential(
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, padding=padding
            )
        )
        self.out_layer=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            norm_layer(output_dim),
        )
        self.roll_out_conv=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            RollOut_Conv(output_dim, output_dim),
        )
        if self.use_attn:
            self.img_sampler = MultiImage_TriFuse_Sampler(inner_channel=output_dim, triplane_in_channels=output_dim,
                                                             img_in_channels=img_in_channels,reso=reso,vit_reso=vit_reso,
                                                             out_channels=output_dim,max_nimg=5,padding=triplane_padding)
        self.down_conv=nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding)

        self.map_layer0 = nn.Linear(in_features=t_input_dim, out_features=output_dim)
        self.map_layer1 = nn.Linear(in_features=output_dim, out_features=output_dim)
    def forward(self, x,t_emb,img_feat,proj_mat,valid_frames):
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        t_emb = t_emb[:,:,None,None]

        out=self.conv_block(x)+t_emb
        out=self.out_layer(out)
        feature=out+self.conv_skip(x)
        feature = self.roll_out_conv(feature)
        if self.use_attn:
            feature=self.img_sampler(feature,img_feat,proj_mat,valid_frames)+feature #skip connect
        feature=self.down_conv(feature)

        return feature


class ResidualConv_GlobalAtten(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, reso=64,
                 vit_reso=16,t_input_dim=256,img_in_channels=1280,use_attn=True,triplane_padding=0.1,
                 norm="batch"):
        super(ResidualConv_GlobalAtten, self).__init__()
        self.use_attn=use_attn

        if norm=="batch":
            norm_layer=nn.BatchNorm2d
        elif norm==None:
            norm_layer=nn.Identity

        self.conv_block = nn.Sequential(
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, padding=padding
            )
        )
        self.out_layer=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            norm_layer(output_dim),
        )
        self.roll_out_conv=nn.Sequential(
            norm_layer(output_dim),
            nn.ReLU(),
            RollOut_Conv(output_dim, output_dim),
        )
        if self.use_attn:
            self.img_sampler = MultiImage_Global_Sampler(inner_channel=output_dim, triplane_in_channels=output_dim,
                                                             img_in_channels=img_in_channels,reso=reso,vit_reso=vit_reso,
                                                             out_channels=output_dim,max_nimg=5,padding=triplane_padding)
        self.down_conv=nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding)

        self.map_layer0 = nn.Linear(in_features=t_input_dim, out_features=output_dim)
        self.map_layer1 = nn.Linear(in_features=output_dim, out_features=output_dim)
    def forward(self, x,t_emb,img_feat,proj_mat,valid_frames):
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        t_emb = t_emb[:,:,None,None]

        out=self.conv_block(x)+t_emb
        out=self.out_layer(out)
        feature=out+self.conv_skip(x)
        feature = self.roll_out_conv(feature)
        if self.use_attn:
            feature=self.img_sampler(feature,img_feat,proj_mat,valid_frames)+feature #skip connect
        feature=self.down_conv(feature)

        return feature

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, t_input_dim=256):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            RollOut_Conv(output_dim,output_dim),
        )
        self.out_layer=nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

        self.map_layer0 = nn.Linear(in_features=t_input_dim, out_features=output_dim)
        self.map_layer1 = nn.Linear(in_features=output_dim, out_features=output_dim)
    def forward(self, x,t_emb):
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        t_emb = t_emb[:,:,None,None]

        out=self.conv_block(x)+t_emb
        out=self.out_layer(out)

        return out + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class ResUnet_Par_cond(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512, 1024],output_channel=32,par_channel=32):
        super(ResUnet_Par_cond, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0]+par_channel, filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ResidualConv(filters[3],filters[4],2,1)


        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0]+par_channel, filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            #nn.LayerNorm(filters[0]),
            nn.LayerNorm(64),#normalize along width dimension, usually it should normalize along channel dimension,
            # I don't know why, but the finetuning performance increase significantly
            zero_module(nn.Conv2d(filters[0], output_channel, 1, 1,bias=False)),
        )
        self.par_channel=par_channel
        self.par_conv=nn.Sequential(
            nn.Conv2d(par_channel, par_channel, kernel_size=3, padding=1),
        )
        self.t_emb_layer=PositionalEmbedding(256)
        self.cat_emb=nn.Linear(
            in_features=6,
            out_features=256,
        )

    def forward(self, x,t,category_code,par_point_feat):
        # Encode
        t_emb=self.t_emb_layer(t)
        cat_emb=self.cat_emb(category_code)
        t_emb=t_emb+cat_emb
        #print(t_emb.shape)
        x1 = self.input_layer(x) + self.input_skip(x)
        if par_point_feat is not None:
            par_point_feat=self.par_conv(par_point_feat)
        else:
            bs,_,H,W=x1.shape
            #print(x1.shape)
            par_point_feat=torch.zeros((bs,self.par_channel,H,W)).float().to(x1.device)
        x1 = torch.cat([x1, par_point_feat], dim=1)
        x2 = self.residual_conv_1(x1,t_emb)
        x3 = self.residual_conv_2(x2,t_emb)
        # Bridge
        x4 = self.residual_conv_3(x3,t_emb)
        x5 = self.bridge(x4,t_emb)

        x6=self.upsample_1(x5)
        x6=torch.cat([x6,x4],dim=1)
        x7=self.up_residual_conv1(x6,t_emb)

        x7=self.upsample_2(x7)
        x7=torch.cat([x7,x3],dim=1)
        x8=self.up_residual_conv2(x7,t_emb)

        x8 = self.upsample_3(x8)
        x8 = torch.cat([x8, x2], dim=1)
        #print(x8.shape)
        x9 = self.up_residual_conv3(x8,t_emb)

        x9 = self.upsample_4(x9)
        x9 = torch.cat([x9, x1], dim=1)
        x10 = self.up_residual_conv4(x9,t_emb)

        output=self.output_layer(x10)

        return output

class ResUnet_DirectAttenMultiImg_Cond(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512, 1024],
                 img_in_channels=1024,vit_reso=16,output_channel=32,
                 use_par=False,par_channel=32,triplane_padding=0.1,norm='batch',
                 use_cat_embedding=False,
                 block_type="multiview_local"):
        super(ResUnet_DirectAttenMultiImg_Cond, self).__init__()

        if block_type == "multiview_local":
            block=ResidualConv_MultiImgAtten
        elif block_type =="multiview_global":
            block=ResidualConv_GlobalAtten
        elif block_type =="multiview_tri":
            block=ResidualConv_TriMultiImgAtten
        else:
            raise NotImplementedError

        if norm=="batch":
            norm_layer=nn.BatchNorm2d
        elif norm==None:
            norm_layer=nn.Identity

        self.use_cat_embedding=use_cat_embedding
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            norm_layer(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        self.use_par=use_par
        input_1_channels=filters[0]
        if self.use_par:
            self.par_conv = nn.Sequential(
                nn.Conv2d(par_channel, par_channel, kernel_size=3, padding=1),
            )
            input_1_channels=filters[0]+par_channel
        self.residual_conv_1 = block(input_1_channels, filters[1], 2, 1,reso=64
                                                          ,use_attn=False,triplane_padding=triplane_padding,norm=norm)
        self.residual_conv_2 = block(filters[1], filters[2], 2, 1, reso=32,
                                                          use_attn=False,triplane_padding=triplane_padding,norm=norm)
        self.residual_conv_3 = block(filters[2], filters[3], 2, 1,reso=16,
                                                          use_attn=False,triplane_padding=triplane_padding,norm=norm)
        self.bridge = block(filters[3] , filters[4], 2, 1, reso=8
                                                 ,use_attn=False,triplane_padding=triplane_padding,norm=norm) #input reso is 8, output reso is 4


        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = block(filters[4] + filters[3], filters[3], 1, 1,reso=8,img_in_channels=img_in_channels,vit_reso=vit_reso,
                                                            use_attn=True,triplane_padding=triplane_padding,norm=norm)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = block(filters[3] + filters[2], filters[2], 1, 1,reso=16,img_in_channels=img_in_channels,vit_reso=vit_reso,
                                                            use_attn=True,triplane_padding=triplane_padding,norm=norm)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = block(filters[2] + filters[1], filters[1], 1, 1,reso=32,img_in_channels=img_in_channels,vit_reso=vit_reso,
                                                            use_attn=True,triplane_padding=triplane_padding,norm=norm)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = block(filters[1] + input_1_channels, filters[0], 1, 1, reso=64,
                                                            use_attn=False,triplane_padding=triplane_padding,norm=norm)

        self.output_layer = nn.Sequential(
            nn.LayerNorm(64), #normalize along width dimension, usually it should normalize along channel dimension,
            # I don't know why, but the finetuning performance increase significantly
            #nn.LayerNorm([filters[0], 192, 64]),
            zero_module(nn.Conv2d(filters[0], output_channel, 1, 1,bias=False)),
        )
        self.t_emb_layer=PositionalEmbedding(256)
        if use_cat_embedding:
            self.cat_emb = nn.Linear(
                in_features=6,
                out_features=256,
            )

    def forward(self, x,t,image_emb,proj_mat,valid_frames,category_code,par_point_feat=None):
        # Encode
        t_emb=self.t_emb_layer(t)
        if self.use_cat_embedding:
            cat_emb=self.cat_emb(category_code)
            t_emb=t_emb+cat_emb
        x1 = self.input_layer(x) + self.input_skip(x)
        if self.use_par:
            par_point_feat=self.par_conv(par_point_feat)
            x1 = torch.cat([x1, par_point_feat], dim=1)
        x2 = self.residual_conv_1(x1,t_emb,image_emb,proj_mat,valid_frames)
        x3 = self.residual_conv_2(x2,t_emb,image_emb,proj_mat,valid_frames)
        x4 = self.residual_conv_3(x3,t_emb,image_emb,proj_mat,valid_frames)
        x5 = self.bridge(x4,t_emb,image_emb,proj_mat,valid_frames)

        x6=self.upsample_1(x5)
        x6=torch.cat([x6,x4],dim=1)
        x7=self.up_residual_conv1(x6,t_emb,image_emb,proj_mat,valid_frames)

        x7=self.upsample_2(x7)
        x7=torch.cat([x7,x3],dim=1)
        x8=self.up_residual_conv2(x7,t_emb,image_emb,proj_mat,valid_frames)

        x8 = self.upsample_3(x8)
        x8 = torch.cat([x8, x2], dim=1)
        #print(x8.shape)
        x9 = self.up_residual_conv3(x8,t_emb,image_emb,proj_mat,valid_frames)

        x9 = self.upsample_4(x9)
        x9 = torch.cat([x9, x1], dim=1)
        x10 = self.up_residual_conv4(x9,t_emb,image_emb,proj_mat,valid_frames)

        output=self.output_layer(x10)

        return output


if __name__=="__main__":
    net=ResUnet(32,output_channel=32).float().cuda()
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model = %s" % str(net))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    par_point_feat=torch.randn((10,32,64*3,64)).float().cuda()
    input=torch.randn((10,32,64*3,64)).float().cuda()
    t=torch.randn((10,1,1,1)).float().cuda()
    output=net(input,t.flatten(),par_point_feat)
    #print(output.shape)