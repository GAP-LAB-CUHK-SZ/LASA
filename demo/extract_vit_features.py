import os,sys
sys.path.append("..")
from simple_dataset import Simple_InTheWild_dataset
import argparse
from torch.utils.data import DataLoader
import timm
import torch
import numpy as np
from process_scripts import misc

parser=argparse.ArgumentParser()
parser.add_argument("--data_dir",type=str,default="../example_process_data")
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--scene_id',default="all",type=str)
args=parser.parse_args()


misc.init_distributed_mode(args)
dataset=Simple_InTheWild_dataset(dataset_dir=args.data_dir,scene_id=args.scene_id,n_px=224)
num_tasks = misc.get_world_size()
global_rank = misc.get_rank()
print(num_tasks,global_rank)
sampler = torch.utils.data.DistributedSampler(
    dataset, num_replicas=num_tasks, rank=global_rank,
    shuffle=False)  # shuffle=True to reduce monitor bias

dataloader=DataLoader(
    dataset,
    sampler=sampler,
    batch_size=10,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)
VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'
model=timm.create_model(VIT_MODEL, pretrained=True,pretrained_cfg_overlay=dict(file="./open_clip_pytorch_model.bin"))
model=model.eval().float().cuda()
for idx,data_batch in enumerate(dataloader):
    if idx%10==0:
        print("{}/{}".format(dataloader.__len__(),idx))
    images = data_batch["images"].cuda().float()
    model_id= data_batch["model_id"]
    image_name=data_batch["image_name"]
    scene_id=data_batch["scene_id"]
    with torch.no_grad():
        output_features=model.forward_features(images)
    for j in range(output_features.shape[0]):
        save_folder=os.path.join(args.data_dir,scene_id[j],"7_img_feature",model_id[j])
        os.makedirs(save_folder,exist_ok=True)
        save_path=os.path.join(save_folder,image_name[j]+".npz")
        np.savez_compressed(save_path,img_features=output_features[j].detach().cpu().numpy().astype(np.float32))
