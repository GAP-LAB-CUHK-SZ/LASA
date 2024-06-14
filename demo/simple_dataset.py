import torch
import torch.nn as nn
from torch.utils import data
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import glob
import numpy as np
import open3d as o3d
import cv2
from datasets.taxonomy import category_map as category_ids

classname_map={
    "chair":["chair","stool"],
    "cabinet":["dishwasher","cabinet","oven","refrigerator",'storage'],
    "sofa":["sofa"],
    "table":["table"],
    "bed":["bed"],
    "shelf":["shelf"]
}
classname_remap={ #map small categories to six large categories
    "chair":"chair",
    "stool":"chair",
    "dishwasher":"cabinet",
    "cabinet":"cabinet",
    "oven":"cabinet",
    "refrigerator":"cabinet",
    "storage":"cabinet",
    "sofa":"sofa",
    "table":"table",
    "bed":"bed",
    "shelf":"shelf"
}

def image_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
    ])
class Simple_InTheWild_dataset(data.Dataset):
    def __init__(self,dataset_dir="/data1/haolin/data/real_scene_process_data",scene_id="letian-310",n_px=224):
        self.dataset_dir=dataset_dir
        self.preprocess = image_transform(n_px)
        self.image_path = []
        if scene_id=="all":
            scene_list=os.listdir(self.dataset_dir)
            for id in scene_list:
                image_folder=os.path.join(self.dataset_dir,id,"6_images")
                self.image_path+=glob.glob(image_folder+"/*/*jpg")
        else:
            image_folder = os.path.join(self.dataset_dir, scene_id, "6_images")
            self.image_path += glob.glob(image_folder + "/*/*jpg")
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        path=self.image_path[index]
        basename=os.path.basename(path)[:-4]
        model_id=path.split(os.sep)[-2]
        scene_id=path.split(os.sep)[-4]
        image=Image.open(path)
        image_tensor=self.preprocess(image)

        return {"images":image_tensor,"image_name":basename,"model_id":model_id,"scene_id":scene_id}

class InTheWild_Dataset(data.Dataset):
    def __init__(self,data_dir="/data1/haolin/data/real_scene_process_data/letian-310",scene_id="letian-310",
                 par_pc_size=2048,category="chair",max_n_imgs=5):
        self.par_pc_size=par_pc_size
        self.data_dir=data_dir
        self.category=category
        self.max_n_imgs=max_n_imgs

        self.models=[]
        category_list=classname_map[category]
        modelid_list=[]
        for cat in category_list:
            if scene_id=="all":
                scene_list=os.listdir(self.data_dir)
                for id in scene_list:
                    data_folder=os.path.join(self.data_dir,id)
                    modelid_list+=glob.glob(data_folder+"/6_images/%s*"%(cat))
            else:
                data_folder=os.path.join(self.data_dir,scene_id)
                modelid_list+=glob.glob(data_folder+"/6_images/%s*"%(cat))
        sceneid_list = [item.split("/")[-3] for item in modelid_list]
        modelid_list=[item.split("/")[-1] for item in modelid_list]
        for idx,modelid in enumerate(modelid_list):
            scene_id=sceneid_list[idx]
            image_folder=os.path.join(self.data_dir,scene_id,"6_images",modelid)
            image_list=os.listdir(image_folder)
            if len(image_list)==0:
                continue
            imageid_list=[item[0:-4] for item in image_list]
            imageid_list.sort(key=lambda x:int(x))
            partial_path=os.path.join(self.data_dir,scene_id,"5_partial_points",modelid+".ply")
            if os.path.exists(partial_path)==False: continue
            self.models+=[
                {'model_id':modelid,
                 "scene_id":scene_id,
                 "partial_path":partial_path,
                 "imageid_list":imageid_list,
                 }
            ]
    def __len__(self):
        return len(self.models)

    def __getitem__(self,idx):
        model = self.models[idx]['model_id']
        scene_id=self.models[idx]['scene_id']
        imageid_list = self.models[idx]['imageid_list']
        partial_path=self.models[idx]['partial_path']
        n_frames=min(len(imageid_list),self.max_n_imgs)
        img_indexes=np.linspace(start=0,stop=len(imageid_list)-1,num=n_frames).astype(np.int32)

        '''load partial points'''
        par_point_o3d = o3d.io.read_point_cloud(partial_path)
        par_points = np.asarray(par_point_o3d.points)
        replace = par_points.shape[0] < self.par_pc_size
        ind = np.random.default_rng().choice(par_points.shape[0], self.par_pc_size, replace=replace)
        par_points=par_points[ind]
        par_points=torch.from_numpy(par_points).float()

        '''load image features'''
        image_list=[]
        valid_frames = []
        image_namelist=[]
        for img_index in img_indexes:
            image_name = imageid_list[img_index]
            image_feat_path = os.path.join(self.data_dir,scene_id, "7_img_feature", model,image_name + '.npz')
            image = np.load(image_feat_path)["img_features"]
            image_list.append(torch.from_numpy(image).float())
            image_namelist.append(image_name)
            valid_frames.append(True)
        '''load original image'''
        org_img_list=[]
        for img_index in img_indexes:
            image_name = imageid_list[img_index]
            image_path = os.path.join(self.data_dir,scene_id, "6_images", model,image_name+".jpg")
            org_image = cv2.imread(image_path)
            org_image = cv2.resize(org_image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            org_img_list.append(org_image)

        '''load project matrix'''
        proj_mat_list=[]
        for img_index in img_indexes:
            image_name = imageid_list[img_index]
            proj_mat_path = os.path.join(self.data_dir,scene_id, "8_proj_matrix", model, image_name + ".npy")
            proj_mat = np.load(proj_mat_path)
            proj_mat_list.append(proj_mat)

        '''load transformation matrix'''
        tran_mat_path = os.path.join(self.data_dir,scene_id, "10_tran_matrix", model+".npy")
        tran_mat = np.load(tran_mat_path)

        '''category code, not used for category specific models'''
        category_id = category_ids[self.category]
        one_hot = torch.zeros((6)).float()
        one_hot[category_id] = 1.0

        ret_dict={
            "model_id":model,
            "scene_id":scene_id,
            "par_points":par_points,
            "proj_mat":torch.stack([torch.from_numpy(mat) for mat in proj_mat_list], dim=0),
            "tran_mat":torch.from_numpy(tran_mat).float(),
            "image":torch.stack(image_list,dim=0),
            "org_image":org_img_list,
            "valid_frames":torch.tensor(valid_frames).bool(),
            "category_ids": category_id,
            "category_code":one_hot,
        }
        return ret_dict

