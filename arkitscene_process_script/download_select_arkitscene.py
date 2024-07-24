import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)

args=parser.parse_args()
save_dir=args.save_dir
with open("annotate_scene_list.txt",'r') as f:
    scene_list=f.readlines()
    scene_list=[scene_id.rstrip("\n") for scene_id in scene_list]

# for scene_id in scene_list:
#     cmd=r"python download_data.py 3dod --video_id %s --split Training --download_dir %s"%(scene_id,save_dir)
#     os.system(cmd)
for scene_id in scene_list:
    cmd=r"python download_data.py 3dod --video_id %s --split Validation --download_dir %s"%(scene_id,save_dir)
    os.system(cmd)

