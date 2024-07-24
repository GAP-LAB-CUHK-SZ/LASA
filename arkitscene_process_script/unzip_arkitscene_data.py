import os
import zipfile
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--arkit_dir",type=str,required=True)
parser.add_argument("--split",type=str,default="Training")
args=parser.parse_args()

data_dir=os.path.join(args.arkit_dir,args.split)
scene_list_path= r"annotate_scene_list.txt"
with open(scene_list_path,'r') as f:
    #print(f)
    scene_list=f.readlines()
scene_list=[item.rstrip("\n") for item in scene_list]

filelist=os.listdir(data_dir)
filelist=[item for item in filelist if ".zip" in item]


for file in filelist[0:]:
    if file[:-4] not in scene_list: #if not inside the annotate list
        continue
    save_folder = data_dir

    src_path=os.path.join(data_dir,file)
    zip=zipfile.ZipFile(src_path)
    namelist=zip.namelist()
    for file in namelist:
        zip.extract(file,save_folder)