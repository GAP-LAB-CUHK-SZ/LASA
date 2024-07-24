import numpy as np
import sys
from arkit_utils.tenFpsDataLoader import TenFpsDataLoader
import os
from arkit_utils.extract_utils import extract_gt
from arkit_utils import rotation
from arkit_utils import box_utils
from arkit_utils.taxonomy import class_names
import pandas as pd
import time
from tqdm import tqdm
import cv2
import argparse
import copy

def rotate_image(img, direction):
    if direction == 'Up':
        pass
    elif direction == 'Left':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'Right':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'Down':
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise Exception(f'No such direction (={direction}) rotation')
    return img

def rotate_image_PIL(img, direction):
    if direction == 'Up':
        pass
    elif direction == 'Left':
        img = img.rotate(270, expand=True)
    elif direction == 'Right':
        img = img.rotate(90, expand=True)
    elif direction == 'Down':
        img = img.rotate(180, expand=True)
    else:
        raise Exception(f'No such direction (={direction}) rotation')
    return img

parser=argparse.ArgumentParser()
parser.add_argument(
    "--arkit_root",
    default=r"E:\ARKitScenes_data\3dod",
    type=str
)
parser.add_argument(
    "--save_root",
    default=r"D:\arkit_scene\images",
    type=str
)
parser.add_argument(
    "--split",
    default="Training",
    type=str,
)
args=parser.parse_args()

scene_id_list=os.listdir(os.path.join(args.arkit_root,args.split))
scene_id_list=[scene_id for scene_id in scene_id_list if os.path.isdir(os.path.join(args.arkit_root,args.split,scene_id))]

with open("annotate_scene_list.txt",'r') as f:
    anno_list=f.readlines()
    anno_list=[item.rstrip("\n") for item in anno_list]

for scene_id in scene_id_list:
    if scene_id not in anno_list:
        continue
    split=args.split
    data_root=os.path.join(args.arkit_root,split)
    save_root=args.save_root

    save_folder=os.path.join(save_root,scene_id)
    os.makedirs(save_folder,exist_ok=True)
    if len(os.listdir(save_folder))>0:
        continue

    gt_fn = os.path.join(data_root, scene_id, f'{scene_id}_3dod_annotation.json')
    skipped, boxes_corners, centers, sizes, labels, uids,bbox_voxels,boxes_mat = extract_gt(gt_fn)
    if skipped or boxes_corners.shape[0] == 0:
        continue
    n_gt = boxes_corners.shape[0]
    label_type = np.array([labels, uids])
    df=pd.read_csv(os.path.join(args.arkit_root,'metadata.csv'))
    sky_direction = df.loc[df['video_id'] == int(scene_id), 'sky_direction'].values[0]

    # step 0.2: data
    data_path = os.path.join(data_root, scene_id, f"{scene_id}_frames")
    loader = TenFpsDataLoader(
        dataset_cfg=None,
        class_names=class_names,
        root_path=data_path,
        frame_rate=3
    )


    t = time.time()

    world_pc, world_rgb = [], []
    total_mask = []
    for i in tqdm(range(len(loader))):
        frame = loader[i]
        image_path = frame["image_path"]
        frame_id = image_path.split(".")[-2]
        frame_id = frame_id.split("_")[-1]

        # step 2.1 get data accumulated to current frame
        # in upright camera coordinate system
        image = frame["image"]
        depth = frame["depth"]
        pose = frame["pose"]
        pcd = frame["pcd"]  # in world coordinate
        rgb = frame["color"]
        intrinsic = frame["intrinsics"]
        urc = np.linalg.inv(pose) #wrd2cam_matrix

        # if args.vis:
        #     # depth_vis = np.clip(depth / 1000, 0, 5)
        #     depth_vis = depth / np.max(depth)
        #     depth_vis = cmap(depth_vis)[..., :3] * 255

        # rotate pcd to urc coordinate
        pcd = rotation.rotate_pc(pcd, urc)

        # 2. gt_boxes
        boxes_corners_urc = rotation.rotate_pc(
            boxes_corners.reshape(-1, 3), urc
        ).reshape(-1, 8, 3)

        bboxes_voxel_urc = rotation.rotate_pc(
            bbox_voxels.reshape(-1,3), urc
        ).reshape(-1,1000,3)

        boxes_mat_urc=[]
        for i in range(boxes_mat.shape[0]):
            box_mat_urc=np.dot(urc,boxes_mat[i])
            boxes_mat_urc.append(copy.deepcopy(box_mat_urc))
        boxes_mat_urc=np.stack(boxes_mat_urc)
        #print(boxes_mat_urc.shape)

        mask_pts_in_box = box_utils.points_in_boxes(pcd, boxes_corners_urc)
        pts_cnt = np.sum(mask_pts_in_box, axis=0)

        mask_box = pts_cnt > 100
        for _id, uid in enumerate(uids):
            if not mask_box[_id]:
                continue
            else:
                label = labels[_id]

                output_object_dir = os.path.join(save_folder, f'{label}_{uid}')
                os.makedirs(output_object_dir, exist_ok=True)

                # Export color
                output_object_color_dir = os.path.join(output_object_dir, 'color')
                os.makedirs(output_object_color_dir, exist_ok=True)
                output_proj_mat_dir= os.path.join(output_object_dir, 'proj_matrix')
                os.makedirs(output_proj_mat_dir, exist_ok=True)

                # 1. get camera intrinsic parameters
                img_h, img_w = image.shape[:2]

                # 2. convert open3d coordinate to opencv coordinate
                _bbox_corners = boxes_corners_urc[_id]
                bbox_corners_cv = _bbox_corners.copy()

                bbox_voxel = bboxes_voxel_urc[_id].copy()
                voxel_2d,_ = cv2.projectPoints(bbox_voxel, (0, 0, 0), (0, 0, 0), intrinsic, None)
                voxel_2d = voxel_2d[:,0]
                inside_voxel_mask=(voxel_2d[:,0]>0)&(voxel_2d[:,1]>0)&(voxel_2d[:,0]<img_w-1)&(voxel_2d[:,1]<img_h-1)
                inside_ratio=np.sum(inside_voxel_mask.astype(np.float32))/bbox_voxel.shape[0]
                if inside_ratio<0.4:
                    continue

                # 3. project 3D bounding box corners onto image plane
                corners_2d, _ = cv2.projectPoints(bbox_corners_cv, (0, 0, 0), (0, 0, 0), intrinsic, None)
                corners_2d = corners_2d[:, 0]

                # 3.5 compute projection matrix
                #print(intrinsic)
                box_mat_urc=copy.deepcopy(boxes_mat_urc[_id])
                K4=np.eye(4)
                K4[0:3,0:3]=intrinsic
                #print(K4)
                proj_mat=np.dot(K4,box_mat_urc)

                # 4. compute 2D bounding box
                bbox_min_x = int(np.clip(np.min(corners_2d[:, 0]), 0, img_w))
                bbox_min_y = int(np.clip(np.min(corners_2d[:, 1]), 0, img_h))
                bbox_max_x = int(np.clip(np.max(corners_2d[:, 0]), 0, img_w))
                bbox_max_y = int(np.clip(np.max(corners_2d[:, 1]), 0, img_h))
                bbox_2d = [bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y]

                # 5. Optional: visualize and save 2D bbox annotation
                x_size=bbox_max_x-bbox_min_x
                y_size=bbox_max_y-bbox_min_y
                if x_size/y_size>2:
                    continue
                if y_size/x_size>2:
                    continue
                if max(x_size,y_size)<64:
                    continue

                cv2.imwrite(os.path.join(output_object_color_dir, f"{frame_id}.png"), image)

                bbox_save_path=os.path.join(output_object_color_dir,f"{frame_id}.npy")
                np.save(bbox_save_path,np.array(bbox_2d).astype(np.int32))

                proj_mat_save_path=os.path.join(output_proj_mat_dir,f"{frame_id}.npy")
                np.save(proj_mat_save_path, proj_mat.astype(np.float32))

                #direction of the image
                direct_path=os.path.join(output_object_dir,"direction.txt")
                with open(direct_path,'w') as f:
                    f.write(sky_direction)


    elapased = time.time() - t
    print("total time: %f sec" % elapased)