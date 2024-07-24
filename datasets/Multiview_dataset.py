import os
import glob
import random

import yaml

import torch
from torch.utils import data

import numpy as np
import json

from PIL import Image

import h5py
import torch.distributed as dist
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import pickle as p
import time
import cv2
from torchvision import transforms
import copy
from datasets.taxonomy import category_map_from_synthetic as category_ids
class Object_Occ(data.Dataset):
    def __init__(self, dataset_folder, split, categories=['03001627', "future_chair", 'ABO_chair'], transform=None,
                 sampling=True,
                 num_samples=4096, return_surface=True, surface_sampling=True, surface_size=2048, replica=16):

        self.pc_size = surface_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'occ_data')
        self.mesh_folder = os.path.join(self.dataset_folder, 'other_data')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.readlines()
            models_c = [item.rstrip('\n') for item in models_c]

            for m in models_c[:]:
                if len(m)<=1:
                    continue
                if m.endswith('.npz'):
                    model_id = m[:-4]
                else:
                    model_id = m
                self.models.append({
                    'category': c, 'model': model_id
                })
        self.replica = replica

    def __getitem__(self, idx):
        if self.replica >= 1:
            idx = idx % len(self.models)
        else:
            random_segment = random.randint(0, int(1 / self.replica) - 1)
            idx = int(random_segment * self.replica * len(self.models) + idx)

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.point_folder, category, model + '.npz')
        # print(point_path)
        try:
            start_t = time.time()
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
            end_t = time.time()
            # print("loading time %f"%(end_t-start_t))
        except Exception as e:
            print(e)
            print(point_path)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()
        # scale=1.0

        if self.return_surface:
            pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')
            with np.load(pc_path) as data:
                try:
                    surface = data['points'].astype(np.float32)
                except:
                    print(pc_path,"has problems")
                    raise AttributeError
                surface = surface * scale
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.sampling:
            '''need to conduct label balancing'''
            vol_ind=np.random.default_rng().choice(vol_points.shape[0], self.num_samples,
                                                       replace=(vol_points.shape[0]<self.num_samples))
            near_ind=np.random.default_rng().choice(near_points.shape[0], self.num_samples,
                                                       replace=(near_points.shape[0]<self.num_samples))
            vol_points=vol_points[vol_ind]
            vol_label=vol_label[vol_ind]
            near_points=near_points[near_ind]
            near_label=near_label[near_ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        tran_mat=np.eye(4)
        if self.transform:
            surface, points, _,_, tran_mat = self.transform(surface, points)

        data_dict = {
            "points": points,
            "labels": labels,
            "category_ids": category_ids[category],
            "model_id": model,
            "tran_mat":tran_mat,
            "category":category,
        }
        if self.return_surface:
            data_dict["surface"] = surface

        return data_dict

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return int(len(self.models) * self.replica)

class Object_PartialPoints_MultiImg(data.Dataset):
    def __init__(self, dataset_folder, split, split_filename, categories=['03001627', 'future_chair', 'ABO_chair'],
                 transform=None, sampling=True, num_samples=4096,
                 return_surface=True, ret_sample=True,surface_sampling=True,
                 surface_size=20000,par_pc_size=2048, par_point_aug=None,par_prefix="aug7_",
                 load_proj_mat=False,load_image=False,load_org_img=False,max_img_length=5,load_triplane=True,replica=2,
                 eval_multiview=False,num_objects=-1):

        self.surface_size = surface_size
        self.par_pc_size=par_pc_size
        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.par_point_aug=par_point_aug
        self.par_prefix=par_prefix

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.ret_sample=ret_sample
        self.surface_sampling = surface_sampling
        self.load_proj_mat=load_proj_mat
        self.load_img=load_image
        self.load_org_img=load_org_img
        self.load_triplane=load_triplane
        self.max_img_length=max_img_length
        self.eval_multiview=eval_multiview

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'occ_data')
        self.mesh_folder = os.path.join(self.dataset_folder, 'other_data')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        print(categories)
        self.models = []
        self.model_images_names = {}
        for c_idx, c in enumerate(categories):
            cat_count=0
            subpath = os.path.join(self.point_folder, c)
            print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split_filename)
            with open(split_file, 'r') as f:
                splits = json.load(f)
            for item in splits:
                # print(item)
                model_id = item['model_id']
                image_filenames = item['image_filenames']
                partial_filenames = item['partial_filenames']
                if len(image_filenames)==0 or len(partial_filenames)==0:
                    continue
                self.model_images_names[model_id] = image_filenames
                if split=="train":
                    self.models += [
                        {'category': c, 'model': model_id, "partial_filenames": partial_filenames,
                         "image_filenames": image_filenames}
                    ]
                else:
                    if self.eval_multiview:
                        for length in range(0,len(image_filenames)):
                            self.models+=[
                                 {'category': c, 'model': model_id, "partial_filenames": partial_filenames[0:1],
                                    "image_filenames": image_filenames[0:length+1]}
                            ]
                    self.models += [
                        {'category': c, 'model': model_id, "partial_filenames": partial_filenames[0:1],
                         "image_filenames": image_filenames}
                    ]
        if num_objects!=-1:
            indexes=np.linspace(0,len(self.models)-1,num=num_objects).astype(np.int32)
            self.models = [self.models[i] for i in indexes]

        self.replica = replica

    def load_samples(self,point_path):
        try:
            start_t = time.time()
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
            end_t = time.time()
            # print("reading time %f"%(end_t-start_t))
        except Exception as e:
            print(e)
            print(point_path)
        return vol_points,vol_label,near_points,near_label

    def load_surface(self,surface_path,scale):
        with np.load(surface_path) as data:
            surface = data['points'].astype(np.float32)
            surface = surface * scale
        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.surface_size, replace=False)
            surface = surface[ind]
        surface = torch.from_numpy(surface).float()
        return surface

    def load_par_points(self,partial_path,scale):
        # print(partial_path)
        par_point_o3d = o3d.io.read_point_cloud(partial_path)
        par_points = np.asarray(par_point_o3d.points)
        par_points = par_points * scale
        replace = par_points.shape[0] < self.par_pc_size
        ind = np.random.default_rng().choice(par_points.shape[0], self.par_pc_size, replace=replace)
        par_points = par_points[ind]
        par_points = torch.from_numpy(par_points).float()
        return par_points

    def process_samples(self,vol_points,vol_label,near_points,near_label):
        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()
        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            ind = np.random.default_rng().choice(vol_points.shape[0], 100000, replace=False)
            points = vol_points[ind]
            labels = vol_label[ind]
        return points,labels

    def __getitem__(self, idx):
        if self.replica >= 1:
            idx = idx % len(self.models)
        else:
            random_segment = random.randint(0, int(1 / self.replica) - 1)
            idx = int(random_segment * self.replica * len(self.models) + idx)
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        #image_filenames = self.model_images_names[model]
        image_filenames = self.models[idx]["image_filenames"]
        if self.split=="train":
            n_frames = np.random.randint(min(2,len(image_filenames)), min(len(image_filenames) + 1, self.max_img_length + 1))
            img_indexes = np.random.choice(len(image_filenames), n_frames,
                                           replace=(n_frames > len(image_filenames))).tolist()
        else:
            if self.eval_multiview:
                '''use all images'''
                n_frames=len(image_filenames)
                img_indexes=[i for i in range(n_frames)]
            else:
                n_frames = min(len(image_filenames),self.max_img_length)
                img_indexes=np.linspace(start=0,stop=len(image_filenames)-1,num=n_frames).astype(np.int32)

        partial_filenames = self.models[idx]['partial_filenames']
        par_index = np.random.choice(len(partial_filenames), 1)[0]
        partial_name = partial_filenames[par_index]

        vol_points,vol_label,near_points,near_label=None,None,None,None
        points,labels=None,None
        point_path = os.path.join(self.point_folder, category, model + '.npz')
        if self.ret_sample:
            vol_points,vol_label,near_points,near_label=self.load_samples(point_path)
            points,labels = self.process_samples(vol_points, vol_label, near_points,near_label)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        surface=None
        pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')
        if self.return_surface:
            surface=self.load_surface(pc_path,scale)

        partial_path = os.path.join(self.mesh_folder, category, "5_partial_points", model, partial_name)
        if self.par_point_aug is not None and random.random()<self.par_point_aug: #add augmentation
            par_aug_path=os.path.join(self.mesh_folder, category, "5_partial_points", model, self.par_prefix+partial_name)
            #print(par_aug_path,os.path.exists(par_aug_path))
            if os.path.exists(par_aug_path):
                partial_path=par_aug_path
            else:
                raise FileNotFoundError
        par_points=self.load_par_points(partial_path,scale)

        image_list=[]
        valid_frames=[]
        image_namelist=[]
        if self.load_img:
            for img_index in img_indexes:
                image_name=image_filenames[img_index]
                image_feat_path=os.path.join(self.mesh_folder,category,"7_img_features",model,image_name[:-4]+'.npz')
                image=np.load(image_feat_path)["img_features"]
                image_list.append(torch.from_numpy(image).float())
                valid_frames.append(True)
                image_namelist.append(image_name)
            while len(image_list)<self.max_img_length:
                image_list.append(torch.from_numpy(np.zeros(image_list[0].shape).astype(np.float32)).float())
                valid_frames.append(False)
        org_img_list=[]
        if self.load_org_img:
            for img_index in img_indexes:
                image_name = image_filenames[img_index]
                image_path = os.path.join(self.mesh_folder, category, "6_images", model,
                                               image_name)
                org_image = cv2.imread(image_path)
                org_image = cv2.resize(org_image,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
                org_img_list.append(org_image)

        proj_mat=None
        proj_mat_list=[]
        if self.load_proj_mat:
            for img_index in img_indexes:
                image_name = image_filenames[img_index]
                proj_mat_path = os.path.join(self.mesh_folder, category, "8_proj_matrix", model, image_name[:-4]+".npy")
                proj_mat=np.load(proj_mat_path)
                proj_mat_list.append(proj_mat)
            while len(proj_mat_list)<self.max_img_length:
                proj_mat_list.append(np.eye(4))
        tran_mat=None
        if self.load_triplane:
            triplane_folder=os.path.join(self.mesh_folder,category,'9_triplane_kl25_64',model)
            triplane_list=os.listdir(triplane_folder)
            select_index=np.random.randint(0,len(triplane_list))
            triplane_path=os.path.join(triplane_folder,triplane_list[select_index])
            #triplane_path=os.path.join(triplane_folder,"triplane_feat_0.npz")
            triplane_content=np.load(triplane_path)
            triplane_mean,triplane_logvar,tran_mat=triplane_content['mean'],triplane_content['logvar'],triplane_content['tran_mat']
            tran_mat=torch.from_numpy(tran_mat).float()

        if self.transform:
            if not self.load_triplane:
                surface, points, par_points,proj_mat,tran_mat = self.transform(surface, points, par_points,proj_mat_list)
                tran_mat=torch.from_numpy(tran_mat).float()
            else:
                surface, points, par_points, proj_mat = self.transform(surface, points, par_points, proj_mat_list,tran_mat)

        category_id=category_ids[category]
        one_hot=torch.zeros((6)).float()
        one_hot[category_id]=1.0
        ret_dict = {
            "category_ids": category_ids[category],
            "category":category,
            "category_code":one_hot,
            "model_id": model,
            "partial_name": partial_name[:-4],
            "class_name": category,
        }
        if tran_mat is not None:
            ret_dict["tran_mat"]=tran_mat
        if self.ret_sample:
            ret_dict["points"]=points
            ret_dict["labels"]=labels
        if self.return_surface:
            ret_dict["surface"] = surface
        ret_dict["par_points"] = par_points
        if self.load_img:
            ret_dict["image"] = torch.stack(image_list,dim=0)
            ret_dict["valid_frames"]= torch.tensor(valid_frames).bool()
        if self.load_org_img:
            ret_dict["org_image"]=org_img_list
            ret_dict["image_namelist"]=image_namelist
        if self.load_proj_mat:
            ret_dict["proj_mat"]=torch.stack([torch.from_numpy(mat) for mat in proj_mat_list],dim=0)
        if self.load_triplane:
            ret_dict['triplane_mean']=torch.from_numpy(triplane_mean).float()
            ret_dict['triplane_logvar'] = torch.from_numpy(triplane_logvar).float()
        return ret_dict

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return int(len(self.models) * self.replica)