import argparse
import os
import trimesh

parser=argparse.ArgumentParser()
parser.add_argument('--lasa_dir',type=str)
parser.add_argument('--num_tasks',type=int,default=1)
parser.add_argument('--task_id',type=int,default=0)
args=parser.parse_args()

lasa_dir=args.lasa_dir

scene_id_list=os.listdir(lasa_dir)
length=len(scene_id_list)
interval=length//args.num_tasks
if args.task_id<args.num_tasks-1:
    scene_id_list=scene_id_list[interval*args.task_id:interval*(args.task_id+1)]
else:
    scene_id_list=scene_id_list[interval*args.task_id:]
for scene_id in scene_id_list:
    scene_folder=os.path.join(lasa_dir,scene_id,"instances")
    if os.path.exists(scene_folder)==False:
        continue
    object_id_list=os.listdir(scene_folder)
    for object_id in object_id_list:
        object_folder = os.path.join(scene_folder, object_id)
        output_path = os.path.join(object_folder, object_id + "_watertight.obj")
        # if os.path.exists(output_path):
        #     continue
        gt_path=os.path.join(object_folder,object_id+"_gt_mesh_2.obj")
        if os.path.exists(gt_path)==False:
            continue
        gt=trimesh.load(gt_path)
        tri_path=os.path.join(object_folder,object_id+"_tri.obj")
        gt.export(tri_path)
        cmd="manifold %s %s 50000"%(tri_path,output_path)
        os.system(cmd)
