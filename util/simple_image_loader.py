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

def image_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        # Normalize((123.675/255.0,116.28/255.0,103.53/255.0),
        #           (58.395/255.0,57.12/255.0,57.375/255.0))
        Normalize((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
        # Normalize((0.5, 0.5, 0.5),
        #           (0.5, 0.5, 0.5)),
    ])

class Image_dataset(data.Dataset):
    def __init__(self,dataset_folder="/data1/haolin/datasets",categories=['03001627'],n_px=224):
        self.dataset_folder=dataset_folder
        self.image_folder=os.path.join(self.dataset_folder,'other_data')
        self.preprocess=image_transform(n_px)
        self.image_path=[]
        for cat in categories:
            subpath=os.path.join(self.image_folder,cat,"6_images")
            model_list=os.listdir(subpath)
            for folder in model_list:
                model_folder=os.path.join(subpath,folder)
                image_list=os.listdir(model_folder)
                for image_filename in image_list:
                    image_filepath=os.path.join(model_folder,image_filename)
                    self.image_path.append(image_filepath)
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        path=self.image_path[index]
        basename=os.path.basename(path)[:-4]
        model_id=path.split(os.sep)[-2]
        category=path.split(os.sep)[-4]
        image=Image.open(path)
        image_tensor=self.preprocess(image)

        return {"images":image_tensor,"image_name":basename,"model_id":model_id,"category":category}

class Image_InTheWild_dataset(data.Dataset):
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

