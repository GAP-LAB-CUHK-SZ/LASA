import os,sys
import numpy as np
import glob
import open3d as o3d
import json
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--cat",required=True,type=str,nargs="+")
parser.add_argument("--root_dir",type=str,default="../data")
args=parser.parse_args()

sdf_folder="occ_data"
other_folder="other_folder"
data_dir=args.root_dir
category=args.cat

category_list=os.listdir(os.path.join(args.root_dir,other_folder))
category_list=[category for category in category_list if "arkit_" not in category]
for category in category_list:
    train_path = os.path.join(data_dir, sdf_folder, category, "train.lst")
    with open(train_path, 'r') as f:
        train_list = f.readlines()
        train_list = [item.rstrip() for item in train_list]
        if ".npz" in train_list[0]:
            train_list = [item[:-4] for item in train_list]
    val_path = os.path.join(data_dir, sdf_folder, category, "val.lst")
    with open(val_path, 'r') as f:
        val_list = f.readlines()
        val_list = [item.rstrip() for item in val_list]
        if ".npz" in val_list[0]:
            val_list = [item[:-4] for item in val_list]

    sdf_dir=os.path.join(data_dir,sdf_folder,category)
    filelist=os.listdir(sdf_dir)
    model_id_list=[item[:-4] for item in filelist if ".npz" in item]

    train_par_img_list=[]
    val_par_img_list=[]
    for model_id in model_id_list:
        image_dir=os.path.join(data_dir,other_folder,category,"6_images",model_id)
        partial_dir=os.path.join(data_dir,other_folder,category,"5_partial_points",model_id)
        if os.path.exists(image_dir)==False and os.path.exists(partial_dir)==False:
            continue
        if os.path.exists(image_dir):
            image_list=glob.glob(image_dir+"/*.jpg")+glob.glob(image_dir+"/*.png")
            image_list=[os.path.basename(image_path) for image_path in image_list]
        else:
            image_list=[]

        if os.path.exists(partial_dir):
            partial_list=glob.glob(partial_dir+"/partial_points_*.ply")
        else:
            partial_list=[]
        partial_valid_list=[]
        for partial_filepath in partial_list:
            par_o3d=o3d.io.read_point_cloud(partial_filepath)
            par_xyz=np.asarray(par_o3d.points)
            if par_xyz.shape[0]>2048:
                partial_valid_list.append(os.path.basename(partial_filepath))
        if len(image_list)==0 and len(partial_valid_list)==0:
            continue
        ret_dict={
            "model_id":model_id,
            "image_filenames":image_list[:],
            "partial_filenames":partial_valid_list[:]
        }
        if model_id in train_list:
            train_par_img_list.append(ret_dict)
        elif model_id in val_list:
            val_par_img_list.append(ret_dict)

    #print(train_par_img_list)
    train_save_path=os.path.join(sdf_dir,"train_par_img.json")
    with open(train_save_path,'w') as f:
        json.dump(train_par_img_list,f,indent=4)

    val_save_path=os.path.join(sdf_dir,"val_par_img.json")
    with open(val_save_path,'w') as f:
        json.dump(val_par_img_list,f,indent=4)