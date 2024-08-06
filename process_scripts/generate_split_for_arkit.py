import os
import numpy as np
import glob
import open3d as o3d
import json
import argparse
import glob

parser=argparse.ArgumentParser()
parser.add_argument("--keyword",default="lowres",type=str)
parser.add_argument("--root_dir",type=str,default="../submodules/DisCo/data")
args=parser.parse_args()

keyword=args.keyword
sdf_folder="occ_data"
other_folder="other_data"
data_dir=args.root_dir

align_dir=os.path.join(args.root_dir,"align_mat_all") # this alignment matrix is aligned from highres scan to lowres scan
# the alignment matrix is still under cleaning, not all the data have proper alignment matrix yet.
align_filelist=glob.glob(align_dir+"/*/*.txt")
valid_model_list=[]
for align_filepath in align_filelist:
    if "-v" in align_filepath:
        align_mat=np.loadtxt(align_filepath)
        if align_mat.shape[0]!=4:
            continue
        model_id=os.path.basename(align_filepath).split("-")[0]
        valid_model_list.append(model_id)

print("there are %d valid lowres models"%(len(valid_model_list)))

category_list=os.listdir(os.path.join(args.root_dir,other_folder))
category_list=[category for category in category_list if "arkit_" in category]
for category in category_list:
    train_path=os.path.join(data_dir,sdf_folder,category,"train.lst")
    with open(train_path,'r') as f:
        train_list=f.readlines()
        train_list=[item.rstrip() for item in train_list]
        if ".npz" in train_list[0]:
            train_list=[item[:-4] for item in train_list]
    val_path=os.path.join(data_dir,sdf_folder,category,"val.lst")
    with open(val_path,'r') as f:
        val_list=f.readlines()
        val_list=[item.rstrip() for item in val_list]
        if ".npz" in val_list[0]:
            val_list=[item[:-4] for item in val_list]


    sdf_dir=os.path.join(data_dir,sdf_folder,category)
    filelist=os.listdir(sdf_dir)
    model_id_list=[item[:-4] for item in filelist if ".npz" in item]

    train_par_img_list=[]
    val_par_img_list=[]
    for model_id in model_id_list:
        if model_id not in valid_model_list:
            continue
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
            partial_list=glob.glob(partial_dir+"/%s_partial_points_*.ply"%(keyword))
        else:
            partial_list=[]
        partial_valid_list=[]
        for partial_filepath in partial_list:
            par_o3d=o3d.io.read_point_cloud(partial_filepath)
            par_xyz=np.asarray(par_o3d.points)
            if par_xyz.shape[0]>2048:
                partial_valid_list.append(os.path.basename(partial_filepath))
        if model_id in val_list:
            if "%s_partial_points_0.ply"%(keyword) in partial_valid_list:
                partial_valid_list=["%s_partial_points_0.ply"%(keyword)]
            else:
                partial_valid_list=[]
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

    train_save_path=os.path.join(sdf_dir,"%s_train_par_img.json"%(keyword))
    with open(train_save_path,'w') as f:
        json.dump(train_par_img_list,f,indent=4)

    val_save_path=os.path.join(sdf_dir,"%s_val_par_img.json"%(keyword))
    with open(val_save_path,'w') as f:
        json.dump(val_par_img_list,f,indent=4)