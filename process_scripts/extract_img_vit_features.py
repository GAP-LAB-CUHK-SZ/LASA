import os,sys
from simple_image_loader import Image_dataset
from torch.utils.data import DataLoader
import timm
import torch
import numpy as np
import argparse
import misc

parser=argparse.ArgumentParser()

parser.add_argument("--root_dir",type=str, default="../data")
parser.add_argument("--ckpt_path",type=str,default="../open_clip_pytorch_model.bin")
parser.add_argument("--batch_size",type=int,default=24)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
args= parser.parse_args()
misc.init_distributed_mode(args)
category_list=os.listdir(os.path.join(args.root_dir,"other_data"))
print("loading dataset")
dataset=Image_dataset(dataset_folder=args.root_dir,categories=category_list,n_px=224)
num_tasks = misc.get_world_size()
global_rank = misc.get_rank()
sampler = torch.utils.data.DistributedSampler(
    dataset, num_replicas=num_tasks, rank=global_rank,
    shuffle=False)  # shuffle=True to reduce monitor bias

dataloader=DataLoader(
    dataset,
    sampler=sampler,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)
print("loading model")
VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'
model=timm.create_model(VIT_MODEL, pretrained=True,pretrained_cfg_overlay=dict(file=args.ckpt_path))
model=model.eval().float().cuda()
save_dir=os.path.join(args.root_dir,"other_data")
for idx,data_batch in enumerate(dataloader):
    if idx%50==0:
        print("{}/{}".format(dataloader.__len__(),idx))
    images = data_batch["images"].cuda().float()
    model_id= data_batch["model_id"]
    image_name=data_batch["image_name"]
    category=data_batch["category"]
    with torch.no_grad():
        #output=model(images,output_hidden_states=True)
        output_features=model.forward_features(images)
    #predict_depth=output.predicted_depth
    #print(predict_depth.shape)
    for j in range(output_features.shape[0]):
        save_folder=os.path.join(save_dir,category[j],"7_img_features",model_id[j])
        os.makedirs(save_folder,exist_ok=True)
        save_path=os.path.join(save_folder,image_name[j]+".npz")
        #print("saving to",save_path)
        np.savez_compressed(save_path,img_features=output_features[j].detach().cpu().numpy().astype(np.float32))