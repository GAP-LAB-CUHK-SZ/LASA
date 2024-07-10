import os,sys
sys.path.append("..")
from configs.config_utils import CONFIG
from models import get_model
import torch
import numpy as np
import open3d as o3d
import timm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from simple_dataset import InTheWild_Dataset,classname_remap,classname_map
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import mcubes
import trimesh
from torch.utils.data import DataLoader

def image_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
    ])

MAX_IMG_LENGTH=5 #take up to 5 images as inputs

ae_paths={
        "chair":"../checkpoint/ae/chair/best-checkpoint.pth",
        "table":"../checkpoint/ae/table/best-checkpoint.pth",
        "cabinet":"../checkpoint/ae/cabinet/best-checkpoint.pth",
        "shelf":"../checkpoint/ae/shelf/best-checkpoint.pth",
        "sofa":"../checkpoint/ae/sofa/best-checkpoint.pth",
        "bed":"../checkpoint/ae/bed/best-checkpoint.pth"
        }
dm_paths={
        "chair":"../checkpoint/finetune_dm/chair/best-checkpoint.pth",
        "table":"../checkpoint/finetune_dm/table/best-checkpoint.pth",
        "cabinet":"../checkpoint/finetune_dm/cabinet/best-checkpoint.pth",
        "shelf":"../checkpoint/finetune_dm/shelf/best-checkpoint.pth",
        "sofa":"../checkpoint/finetune_dm/sofa/best-checkpoint.pth",
        "bed":"../checkpoint/finetune_dm/bed/best-checkpoint.pth"
        }

def inference(ae_model,dm_model,data_batch,device,reso=256):
    density = reso
    gap = 2.2 / density
    x = np.linspace(-1.1, 1.1, int(density + 1))
    y = np.linspace(-1.1, 1.1, int(density + 1))
    z = np.linspace(-1.1, 1.1, int(density + 1))
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device,
                                                                                                            non_blocking=True)
    with torch.no_grad():
        sample_input = dm_model.prepare_sample_data(data_batch)
        sampled_array = dm_model.sample(sample_input, num_steps=36).float()
        sampled_array = torch.nn.functional.interpolate(sampled_array, scale_factor=2, mode="bilinear")

    model_ids = data_batch['model_id']
    tran_mats = data_batch['tran_mat']

    output_meshes={}

    for j in range(sampled_array.shape[0]):
        grid_list = torch.split(grid, 128 ** 3, dim=1)
        output_list = []
        with torch.no_grad():
            for sub_grid in grid_list:
                output_list.append(ae_model.decode(sampled_array[j:j + 1], sub_grid))
        output = torch.cat(output_list, dim=1)
        logits = output[j].detach()

        volume = logits.view(density + 1, density + 1, density + 1).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)

        verts *= gap
        verts -= 1.1

        tran_mat = tran_mats[j].numpy()
        verts_homo = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
        verts_inwrd = np.dot(verts_homo, tran_mat.T)[:, 0:3]
        m_inwrd = trimesh.Trimesh(verts_inwrd, faces[:, ::-1]) #transform the mesh into world coordinate

        output_meshes[model_ids[j]]=m_inwrd
    return output_meshes

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../example_process_data")
    parser.add_argument('--scene_id', default="example_1", type=str)
    parser.add_argument("--save_dir", type=str,default="../example_output_data")
    args = parser.parse_args()

    config_path="../configs/finetune_triplane_diffusion.yaml"
    config=CONFIG(config_path).config

    '''creating save folder'''
    save_folder=os.path.join(args.save_dir,args.scene_id)
    os.makedirs(save_folder,exist_ok=True)

    '''prepare model'''
    device=torch.device("cuda")
    ae_config=config['model']['ae']
    dm_config=config['model']['dm']
    dm_model=get_model(dm_config).to(device)
    ae_model=get_model(ae_config).to(device)
    dm_model.eval()
    ae_model.eval()

    '''preparing data'''
    '''find out how many classes are there in the whole scene'''
    images_folder=os.path.join(args.data_dir,args.scene_id,"6_images")
    object_id_list=os.listdir(images_folder)
    object_class_list=[item.split("_")[0] for item in object_id_list]
    all_object_class=list(set(object_class_list))

    exist_super_categories=[]
    for object_class in all_object_class:
        if object_class not in classname_remap:
            continue
        else:
            exist_super_categories.append(classname_remap[object_class]) #find which category specific models should be employed
    exist_super_categories=list(set(exist_super_categories))
    for super_category in exist_super_categories:
        print("processing %s"%(super_category))
        ae_ckpt=torch.load(ae_paths[super_category],map_location="cpu")["model"]
        dm_ckpt=torch.load(dm_paths[super_category],map_location="cpu")["model"]
        ae_model.load_state_dict(ae_ckpt)
        dm_model.load_state_dict(dm_ckpt)
        dataset = InTheWild_Dataset(data_dir=args.data_dir, scene_id=args.scene_id, category=super_category, max_n_imgs=5)
        dataloader=DataLoader(
            dataset=dataset,
            num_workers=1,
            batch_size=1,
            shuffle=False
        )
        for data_batch in dataloader:
            output_meshes=inference(ae_model,dm_model,data_batch,device)
            #print(output_meshes)
            for model_id in output_meshes:
                mesh=output_meshes[model_id]
                save_path=os.path.join(save_folder,model_id+".ply")
                print("saving to %s"%(save_path))
                mesh.export(save_path)


