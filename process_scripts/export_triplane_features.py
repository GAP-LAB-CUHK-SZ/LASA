import argparse
import math
import sys
sys.path.append("..")
import numpy as np
import os
import torch

import trimesh

from datasets import Object_Occ,Scale_Shift_Rotate
from models import get_model
from pathlib import Path
import open3d as o3d
from configs.config_utils import CONFIG
import tqdm
from util import misc
from datasets.taxonomy import synthetic_arkit_category_combined

if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--configs',type=str,required=True)
    parser.add_argument('--ae-pth',type=str)
    parser.add_argument("--category",nargs='+', type=str)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data-pth",default="../data",type=str)

    args = parser.parse_args()
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    config_path=args.configs
    config=CONFIG(config_path)
    dataset_config=config.config['dataset']
    dataset_config['data_path']=args.data_pth
    #transform = AxisScaling((0.75, 1.25), True)
    transform=Scale_Shift_Rotate(rot_shift_surface=True,use_scale=True)
    if len(args.category)==1 and args.category[0]=="all":
        category=synthetic_arkit_category_combined["all"]
    else:
        category=args.category
    train_dataset = Object_Occ(dataset_config['data_path'], split="train",
                                categories=category,
                                transform=transform, sampling=True,
                                num_samples=1024, return_surface=True,
                                surface_sampling=True, surface_size=dataset_config['surface_size'],replica=1)
    val_dataset = Object_Occ(dataset_config['data_path'], split="val",
                             categories=category,
                             transform=transform, sampling=True,
                             num_samples=1024, return_surface=True,
                             surface_sampling=True, surface_size=dataset_config['surface_size'],replica=1)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank,
        shuffle=False)  # shuffle=True to reduce monitor bias
    val_sampler=torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank,
        shuffle=False)  # shu
    #dataset=val_dataset
    batch_size=args.batch_size
    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,sampler=train_sampler,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
        drop_last=False,
    )
    dataloader_list=[train_dataloader,val_dataloader]
    #dataloader_list=[val_dataloader]
    output_dir=os.path.join(dataset_config['data_path'],"other_data")
    #output_dir="/data1/haolin/datasets/ShapeNetV2_watertight"

    model_config=config.config['model']
    model=get_model(model_config)
    model.load_state_dict(torch.load(args.ae_pth)['model'])
    model.eval().float().to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    with torch.no_grad():
        for e in range(5):
            for dataloader in dataloader_list:
                for data_iter_step, data_batch in tqdm.tqdm(enumerate(dataloader)):
                    surface = data_batch['surface'].to(device, non_blocking=True)
                    model_ids=data_batch['model_id']
                    tran_mats=data_batch['tran_mat']
                    categories=data_batch['category']
                    with torch.no_grad():
                        plane_feat,_,means,logvars=model.encode(surface)
                        plane_feat=torch.nn.functional.interpolate(plane_feat,scale_factor=0.5,mode='bilinear')
                        vars=torch.exp(logvars)
                        means=torch.nn.functional.interpolate(means,scale_factor=0.5,mode="bilinear")
                        vars=torch.nn.functional.interpolate(vars,scale_factor=0.5,mode="bilinear")/4
                        sample_logvars=torch.log(vars)

                    for j in range(means.shape[0]):
                        #plane_dist=plane_feat[j].float().cpu().numpy()
                        mean=means[j].float().cpu().numpy()
                        logvar=sample_logvars[j].float().cpu().numpy()
                        tran_mat=tran_mats[j].float().cpu().numpy()

                        output_folder=os.path.join(output_dir,categories[j],'9_triplane_kl25_64',model_ids[j])
                        Path(output_folder).mkdir(parents=True, exist_ok=True)
                        exist_len=len(os.listdir(output_folder))
                        save_filepath=os.path.join(output_folder,"triplane_feat_%d.npz"%(exist_len))
                        np.savez_compressed(save_filepath,mean=mean,logvar=logvar,tran_mat=tran_mat)
