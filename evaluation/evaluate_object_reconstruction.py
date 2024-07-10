import argparse
import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np

import mcubes
import os
import torch

import trimesh

from datasets.Multiview_dataset import Object_PartialPoints_MultiImg
from datasets.transforms import Scale_Shift_Rotate
from models import get_model
from pathlib import Path
import open3d as o3d
from configs.config_utils import CONFIG
import cv2
from util.misc import MetricLogger
import scipy
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
from util.projection_utils import draw_proj_image
from util import misc
import time
dist_chamfer=ChamferDistance()


def pc_metrics(p1, p2, space_ext=2, fscore_param=0.01, scale=.5):
    """ p2: reference ponits
        (B, N, 3)
    """
    p1, p2, space_ext = p1 * scale, p2 * scale, space_ext * scale
    f_thresh = space_ext * fscore_param

    #print(p1.shape,p2.shape)
    d1, d2, _, _ = dist_chamfer(p1, p2)
    #print(d1.shape,d2.shape)
    d1sqrt, d2sqrt = (d1 ** .5), (d2 ** .5)
    chamfer_L1 = d1sqrt.mean(axis=-1) + d2sqrt.mean(axis=-1)
    chamfer_L2 = d1.mean(axis=-1) + d2.mean(axis=-1)
    precision = (d1sqrt < f_thresh).sum(axis=-1).float() / p1.shape[1]
    recall = (d2sqrt < f_thresh).sum(axis=-1).float() / p2.shape[1]
    #print(precision,recall)
    fscore = 2 * torch.div(recall * precision, recall + precision)
    fscore[fscore == float("inf")] = 0
    return chamfer_L1,chamfer_L2,fscore

if __name__ == "__main__":

    parser = argparse.ArgumentParser('this script can be used to compute iou fscore chamfer distance before icp align', add_help=False)
    parser.add_argument('--configs',type=str,required=True)
    parser.add_argument('--output_folder', type=str, default="../output_result/Triplane_diff_parcond_0926")
    parser.add_argument('--dm-pth',type=str)
    parser.add_argument('--ae-pth',type=str)
    parser.add_argument('--data-pth', type=str,default="../")
    parser.add_argument('--save_mesh',action="store_true",default=False)
    parser.add_argument('--save_image',action="store_true",default=False)
    parser.add_argument('--save_par_points', action="store_true", default=False)
    parser.add_argument('--save_proj_img',action="store_true",default=False)
    parser.add_argument('--save_surface',action="store_true",default=False)
    parser.add_argument('--reso',default=128,type=int)
    parser.add_argument('--category',nargs="+",type=str)
    parser.add_argument('--eval_cd',action="store_true",default=False)
    parser.add_argument('--use_augmentation',action="store_true",default=False)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    misc.init_distributed_mode(args)
    config_path=args.configs
    config=CONFIG(config_path)
    dataset_config=config.config['dataset']
    dataset_config['data_path']=args.data_pth
    if "arkit" in args.category[0]:
        split_filename=dataset_config['keyword']+'_val_par_img.json'
    else:
        split_filename='val_par_img.json'

    transform = None
    if args.use_augmentation:
        transform=Scale_Shift_Rotate(jitter_partial=False,jitter=False,use_scale=False,angle=(-10,10),shift=(-0.1,0.1))
    dataset_val = Object_PartialPoints_MultiImg(dataset_config['data_path'], split_filename=split_filename,categories=args.category,split="val",
                                transform=transform, sampling=False,
                                num_samples=1024, return_surface=True,ret_sample=True,
                                surface_sampling=True, par_pc_size=dataset_config['par_pc_size'],surface_size=100000,
                                load_proj_mat=True,load_image=True,load_org_img=True,load_triplane=None,par_point_aug=None,replica=1)
    batch_size=1

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    val_sampler = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank,
        shuffle=False)  # shu
    dataloader_val=torch.utils.data.DataLoader(
        dataset_val,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )
    output_folder=args.output_folder

    device = torch.device('cuda')

    ae_config=config.config['model']['ae']
    dm_config=config.config['model']['dm']
    ae_model=get_model(ae_config).to(device)
    if args.category[0] == "all":
        dm_config["use_cat_embedding"]=True
    else:
        dm_config["use_cat_embedding"] = False
    dm_model=get_model(dm_config).to(device)
    ae_model.eval()
    dm_model.eval()
    ae_model.load_state_dict(torch.load(args.ae_pth)['model'])
    dm_model.load_state_dict(torch.load(args.dm_pth)['model'])

    density = args.reso
    gap = 2.2 / density
    x = np.linspace(-1.1, 1.1, int(density + 1))
    y = np.linspace(-1.1, 1.1,  int(density + 1))
    z = np.linspace(-1.1, 1.1,  int(density + 1))
    xv, yv, zv = np.meshgrid(x, y, z,indexing='ij')
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device,non_blocking=True)

    metric_logger=MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for data_batch in metric_logger.log_every(dataloader_val,10, header):
            # if data_iter_step==100:
            #     break
            partial_name = data_batch['partial_name']
            class_name = data_batch['class_name']
            model_ids=data_batch['model_id']
            surface=data_batch['surface']
            proj_matrices=data_batch['proj_mat']
            sample_points=data_batch["points"].cuda().float()
            labels=data_batch["labels"].cuda().float()
            sample_input=dm_model.prepare_sample_data(data_batch)
            #t1 = time.time()
            sampled_array = dm_model.sample(sample_input,num_steps=36).float()
            #t2 = time.time()
            #sample_time = t2 - t1
            #print("sampling time %f" % (sample_time))
            sampled_array = torch.nn.functional.interpolate(sampled_array, scale_factor=2, mode="bilinear")
            for j in range(sampled_array.shape[0]):
                if args.save_mesh | args.save_par_points | args.save_image:
                    object_folder = os.path.join(output_folder, class_name[j], model_ids[j])
                    Path(object_folder).mkdir(parents=True, exist_ok=True)
                '''calculate iou'''
                sample_point=sample_points[j:j+1]
                sample_output=ae_model.decode(sampled_array[j:j + 1],sample_point)
                sample_pred=torch.zeros_like(sample_output)
                sample_pred[sample_output>=0.0]=1
                label=labels[j:j+1]
                intersection = (sample_pred * label).sum(dim=1)
                union = (sample_pred + label).gt(0).sum(dim=1)
                iou = intersection * 1.0 / union + 1e-5
                iou = iou.mean()
                metric_logger.update(iou=iou.item())

                if args.use_augmentation:
                    tran_mat=data_batch["tran_mat"][j].numpy()
                    mat_save_path='{}/tran_mat.npy'.format(object_folder)
                    np.save(mat_save_path,tran_mat)

                if args.eval_cd:
                    grid_list=torch.split(grid,128**3,dim=1)
                    output_list=[]
                    #t3=time.time()
                    for sub_grid in grid_list:
                        output_list.append(ae_model.decode(sampled_array[j:j + 1],sub_grid))
                    output=torch.cat(output_list,dim=1)
                    #t4=time.time()
                    #decoding_time=t4-t3
                    #print("decoding time:",decoding_time)
                    logits = output[j].detach()

                    volume = logits.view(density + 1, density + 1, density + 1).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    verts *= gap
                    verts -= 1.1
                    #print("vertice max min",np.amin(verts,axis=0),np.amax(verts,axis=0))


                    m = trimesh.Trimesh(verts, faces)
                    '''calculate fscore and chamfer distance'''
                    result_surface,_=trimesh.sample.sample_surface(m,100000)
                    gt_surface=surface[j]
                    assert gt_surface.shape[0]==result_surface.shape[0]

                    result_surface_gpu = torch.from_numpy(result_surface).float().cuda().unsqueeze(0)
                    gt_surface_gpu = gt_surface.float().cuda().unsqueeze(0)
                    _,chamfer_L2,fscore=pc_metrics(result_surface_gpu,gt_surface_gpu)
                    metric_logger.update(chamferl2=chamfer_L2*1000.0)
                    metric_logger.update(fscore=fscore)

                    if args.save_mesh:
                        m.export('{}/{}_mesh.ply'.format(object_folder, partial_name[j]))

                if args.save_par_points:
                    par_point_input = data_batch['par_points'][j].numpy()
                    #print("input max min", np.amin(par_point_input, axis=0), np.amax(par_point_input, axis=0))
                    par_point_o3d = o3d.geometry.PointCloud()
                    par_point_o3d.points = o3d.utility.Vector3dVector(par_point_input[:, 0:3])
                    o3d.io.write_point_cloud('{}/{}.ply'.format(object_folder, partial_name[j]), par_point_o3d)
                if args.save_image:
                    image_list=data_batch["org_image"]
                    for idx,image in enumerate(image_list):
                        image=image[0].numpy().astype(np.uint8)
                        if args.save_proj_img:
                            proj_mat=proj_matrices[j,idx].numpy()
                            proj_image=draw_proj_image(image,proj_mat,result_surface)
                            proj_save_path = '{}/proj_{}.jpg'.format(object_folder, idx)
                            cv2.imwrite(proj_save_path,proj_image)
                        save_path='{}/{}.jpg'.format(object_folder, idx)
                        cv2.imwrite(save_path,image)
                if args.save_surface:
                    surface=gt_surface.numpy().astype(np.float32)
                    surface_o3d = o3d.geometry.PointCloud()
                    surface_o3d.points = o3d.utility.Vector3dVector(surface[:, 0:3])
                    o3d.io.write_point_cloud('{}/surface.ply'.format(object_folder), surface_o3d)
        metric_logger.synchronize_between_processes()
        print('* iou {ious.global_avg:.3f}'
              .format(ious=metric_logger.iou))
        if args.eval_cd:
            print('* chamferl2 {chamferl2s.global_avg:.3f}'
                  .format(chamferl2s=metric_logger.chamferl2))
            print('* fscore {fscores.global_avg:.3f}'
                  .format(fscores=metric_logger.fscore))
