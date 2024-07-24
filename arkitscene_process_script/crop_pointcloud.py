import os,sys
sys.path.append("..")
import numpy as np
import trimesh
from arkit_utils.extract_utils import extract_gt
from arkit_utils.box_utils import corners_to_boxes,boxes_to_corners_3d


def transform_object_points(obj_points, bbox_param):
    '''Revert bbox points to canonical'''
    center = bbox_param[:3]
    orientation = -bbox_param[6]
    sizes = bbox_param[3:6]

    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    obj_points = (obj_points - center).dot(np.linalg.inv(axis_rectified))
    obj_points_normalize = obj_points / (max(sizes) / 2)
    return obj_points, obj_points_normalize

def rotate_points_along_z(points, angle):
    """Rotation clockwise
    Args:
        points: np.array of np.array (B, N, 3 + C) or
            (N, 3 + C) for single batch
        angle: np.array of np.array (B, )
            or (, ) for single batch
            angle along z-axis, angle increases x ==> y
    Returns:
        points_rot:  (B, N, 3 + C) or (N, 3 + C)

    """
    single_batch = len(points.shape) == 2
    if single_batch:
        points = np.expand_dims(points, axis=0)
        angle = np.expand_dims(angle, axis=0)
    cosa = np.expand_dims(np.cos(angle), axis=1)
    sina = np.expand_dims(np.sin(angle), axis=1)
    zeros = np.zeros_like(cosa) # angle.new_zeros(points.shape[0])
    ones = np.ones_like(sina) # angle.new_ones(points.shape[0])

    rot_matrix = (
        np.concatenate((cosa, -sina, zeros, sina, cosa, zeros, zeros, zeros, ones), axis=1)
        .reshape(-1, 3, 3)
    )

    # print(rot_matrix.view(3, 3))
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    if single_batch:
        points_rot = points_rot.squeeze(0)

    return points_rot

def points_in_boxes(points, boxes):
    """
    Args:
        pc: np.array (n, 3+d)
        boxes: np.array (m, 8, 3)
    Returns:
        mask: np.array (n, m) of type bool
    """
    if len(boxes) == 0:
        return np.zeros([points.shape[0], 1], dtype=np.bool)
    points = points[:, :3]  # get xyz
    # u = p6 - p5
    u = boxes[:, 6, :] - boxes[:, 5, :]  # (m, 3)
    # v = p6 - p7
    v = boxes[:, 6, :] - boxes[:, 7, :]  # (m, 3)
    # w = p6 - p2
    w = boxes[:, 6, :] - boxes[:, 2, :]  # (m, 3)

    # ux, vx, wx
    ux = np.matmul(points, u.T)  # (n, m)
    vx = np.matmul(points, v.T)
    wx = np.matmul(points, w.T)

    # up6, up5, vp6, vp7, wp6, wp2
    up6 = np.sum(u * boxes[:, 6, :], axis=1)
    up5 = np.sum(u * boxes[:, 5, :], axis=1)
    vp6 = np.sum(v * boxes[:, 6, :], axis=1)
    vp7 = np.sum(v * boxes[:, 7, :], axis=1)
    wp6 = np.sum(w * boxes[:, 6, :], axis=1)
    wp2 = np.sum(w * boxes[:, 2, :], axis=1)

    mask_u = np.logical_and(ux <= up6, ux >= up5)  # (1024, n)
    mask_v = np.logical_and(vx <= vp6, vx >= vp7)
    mask_w = np.logical_and(wx <= wp6, wx >= wp2)

    mask = mask_u & mask_v & mask_w  # (10240, n)

    return mask

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lasa_dir", type=str)
args=parser.parse_args()
data_dir=args.lasa_dir
scene_list=os.listdir(data_dir)


for scene_id in scene_list:
    scene_folder=os.path.join(data_dir,scene_id)
    bbox_path=os.path.join(scene_folder,scene_id+"_bbox.npy")
    if os.path.exists(bbox_path)==False:
        continue
    bbox_data=np.load(bbox_path,allow_pickle=True).item()
    #print(bbox_data)
    bboxes=bbox_data["bboxes"]
    class_name=bbox_data["types"]
    uids=bbox_data["uids"]

    bboxes[:,3:6]*=1.2 #enlarge the bbox by 1.2
    bboxes_corner=boxes_to_corners_3d(bboxes)

    laser_path=os.path.join(scene_folder,scene_id+"_faro_aligned_clean_0.04.ply")
    rgbd_path=os.path.join(scene_folder,scene_id+"_arkit_mesh.ply")

    if os.path.exists(laser_path)==False or os.path.exists(rgbd_path)==False:
        continue
    laser_mesh=trimesh.load(laser_path)
    rgbd_mesh=trimesh.load(rgbd_path)

    laser_vert,laser_color=np.asarray(laser_mesh.vertices),np.asarray(laser_mesh.visual.vertex_colors)
    rgbd_vert,rgbd_color=np.asarray(rgbd_mesh.vertices),np.asarray(rgbd_mesh.visual.vertex_colors)

    if os.path.exists(os.path.join(scene_folder,"instances"))==False:
        continue
    object_id_list = os.listdir(os.path.join(scene_folder, "instances"))
    for object_id in object_id_list:
        object_folder=os.path.join(scene_folder,"instances",object_id)
        if os.path.exists(os.path.join(object_folder,"%s_gt_mesh_2.obj"%(object_id)))==False:
            continue
        #laser_save_path=os.path.join(scene_folder,"instances",object_id,"%s_laser_pcd.ply"%(object_id))
        #if os.path.exists(laser_save_path):
        #    continue

        uid=object_id.split("_")[-1]
        index=uids.index(uid)
        bbox_corner=bboxes_corner[index:index+1]

        laser_mask=points_in_boxes(laser_vert,bbox_corner)[:,0]
        laser_object_point=laser_vert[laser_mask>0]

        canonical_laser_points, _ = transform_object_points(laser_object_point, bboxes[index])

        rgbd_mask=points_in_boxes(rgbd_vert,bbox_corner)[:,0]
        rgbd_point=rgbd_vert[rgbd_mask>0]
        canonical_rgbd_points, _ = transform_object_points(rgbd_point, bboxes[index])

        save_folder=os.path.join(data_dir,scene_id,"instances",object_id)
        #print("saving to %s"%(save_folder))
        rgbd_save_path=os.path.join(save_folder,"%s_rgbd_mesh.ply"%(object_id))
        laser_save_path=os.path.join(save_folder,"%s_laser_pcd.ply"%(object_id))

        canonical_laser_points=canonical_laser_points[:,[1,2,0]]
        laser_objects = trimesh.Trimesh(vertices=canonical_laser_points,
                                          vertex_colors=laser_color[laser_mask > 0])
        laser_objects.export(laser_save_path)

        canonical_rgbd_points=canonical_rgbd_points[:,[1,2,0]]
        print("saving to %s"%(rgbd_save_path))
        rgbd_objects = trimesh.Trimesh(vertices=canonical_rgbd_points, vertex_colors=rgbd_color[rgbd_mask > 0])
        rgbd_objects.export(rgbd_save_path)
