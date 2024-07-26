import plyfile
import numpy as np
from argparse import ArgumentParser
import json
import open3d as o3d
import os
import math
import os
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import shutil
import cv2
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

target_width=640
target_height=480

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data', help='path to data dir')
    parser.add_argument('--room_json', help='path to roomplan json')
    parser.add_argument('--output-dir', help='path to output dir')
    parser.add_argument('--max-size', default=-1, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    return args

def scale_camera(cam, target_sizes, ori_sizes):
    """ resize input in order to produce sampled depth map """
    scales = (target_sizes[0] / ori_sizes[0], target_sizes[1] / ori_sizes[1])
    new_cam = np.copy(cam)
    # focal:
    new_cam[0][0] = cam[0][0] * scales[0]
    new_cam[1][1] = cam[1][1] * scales[1]
    # principle point:
    new_cam[0][2] = cam[0][2] * scales[0]
    new_cam[1][2] = cam[1][2] * scales[1]
    return new_cam
    

def write_depth_img(filename, depth):
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(filename)

def to4x4(pose):
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    constants = np.zeros_like(pose[..., :1, :])
    constants[..., :, 3] = 1
    return np.concatenate([pose, constants], axis=-2)

def get_world_points(depth, intrinsics, extrinsics):
    '''
    Args:
        depthmap: H*W
        intrinsics: 3*3 or 4*4
        extrinsics: 4*4, world to camera
    Return:
        points: N*3, in world space 
    '''
    if intrinsics.shape[0] ==4:
        intrinsics = intrinsics[:3,:3]
        
    height, width = depth.shape

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # valid_points = np.ma.masked_greater(depth, 0.0).mask
    # x, y, depth = x[valid_points], y[valid_points], depth[valid_points]

    x = x.reshape((1, height*width))
    y = y.reshape((1, height*width))
    depth = depth.reshape((1, height*width))

    xyz_ref = np.matmul(np.linalg.inv(intrinsics),
                        np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(extrinsics),
                            np.vstack((xyz_ref, np.ones_like(x))))[:3]
    xyz_world = xyz_world.transpose((1, 0))

    return xyz_world


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])
import torch
import torch.nn.functional as F

def get_cam_points(depth, intrinsics):
    '''
    Args:
        depthmap: H*W
        intrinsics: 3*3 or 4*4
        extrinsics: 4*4, world to camera
    Return:
        points: N*3, in camera space 
    '''
    if intrinsics.shape[0] ==4:
        intrinsics = intrinsics[:3,:3]
        
    height, width = depth.shape

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # valid_points = np.ma.masked_greater(depth, 0.0).mask
    # x, y, depth = x[valid_points], y[valid_points], depth[valid_points]

    x = x.reshape((1, height*width))
    y = y.reshape((1, height*width))
    depth = depth.reshape((1, height*width))

    xyz_cam = np.matmul(np.linalg.inv(intrinsics),
                        np.vstack((x, y, np.ones_like(x))) * depth).transpose((1, 0))

    return xyz_cam

def depth2norm(cam_points, height, width, nei=3):
    '''
    Args:
        cam_points: N*3, in camera space 
        height: 1
        weight: 1
    Return:
        
    '''
    pts_3d_map = cam_points.view(-1, height, width, 3)

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    diff_x0 = diff_x0.reshape(-1, 3)
    diff_y0 = diff_y0.reshape(-1, 3)
    diff_x1 = diff_x1.reshape(-1, 3)
    diff_y1 = diff_y1.reshape(-1, 3)
    diff_x0y0 = diff_x0y0.reshape(-1, 3)
    diff_x0y1 = diff_x0y1.reshape(-1, 3)
    diff_x1y0 = diff_x1y0.reshape(-1, 3)
    diff_x1y1 = diff_x1y1.reshape(-1, 3)

    ## calculate normal by cross product of two vectors
    normals0 = torch.cross(diff_x1, diff_y1)
    normals1 =  torch.cross(diff_x0, diff_y0)
    normals2 = torch.cross(diff_x0y1, diff_x0y0)
    normals3 = torch.cross(diff_x1y0, diff_x1y1)

    normal_vector = normals0 + normals1 + normals2 + normals3
    normal_vectorl2 = torch.norm(normal_vector, p=2, dim = 1)
    normal_vector = torch.div(normal_vector.permute(1,0), normal_vectorl2)
    normal_vector = normal_vector.permute(1,0).view(pts_3d_map_ctr.shape).permute(0,3,1,2)
    normal_map = F.pad( normal_vector, (0,2*nei,2*nei,0),"constant",value=0)
    normal = - F.normalize(normal_map, dim=1, p=2)
    return normal

def get_intrinsic_matrices(fx,fy,cx,cy):
    """Returns the intrinsic matrices for each camera.

    Returns:
        Pinhole camera intrinsics matrices
    """
    K = np.zeros((3, 3))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotz(t):
    ''' 3D Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def roty(t):
    ''' 3D Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

if __name__ == "__main__":
    args = parse_args
    source = args.data
    output_dir = os.path.join(args.output_dir, 'extracted')

    os.makedirs(f'{output_dir}/color', exist_ok=True)
    os.makedirs(f'{output_dir}/intrinsic', exist_ok=True)
    os.makedirs(f'{output_dir}/pose', exist_ok=True)
    os.makedirs(f'{output_dir}/depth', exist_ok=True)

    target_image_dir = Path(f'{output_dir}/color')
    target_depth_dir = Path(f'{output_dir}/depth')
    
    # Load image and depth filename
    if (Path(source) / "keyframes" / "corrected_images").exists():
        polycam_image_dir = Path(source) / "keyframes" / "corrected_images"
        polycam_cameras_dir = Path(source) / "keyframes" / "corrected_cameras"
    else:
        polycam_image_dir = Path(source) / "keyframes" / "images"
        polycam_cameras_dir = Path(source)  / "keyframes" / "cameras"
    polycam_depth_dir = Path(source) / "keyframes" / "depth"

    if not polycam_image_dir.exists():
            raise ValueError(f"Image directory {polycam_image_dir} doesn't exist")

    polycam_image_filenames = []
    polycam_depth_filenames = []
    for f in polycam_image_dir.iterdir():
        if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
            polycam_image_filenames.append(f)
    for f in polycam_depth_dir.iterdir():
        if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
            polycam_depth_filenames.append(f)

    # Sample max_dataset_size images
    polycam_image_filenames = sorted(polycam_image_filenames, key=lambda fn: int(fn.stem))
    polycam_depth_filenames = sorted(polycam_depth_filenames, key=lambda fn: int(fn.stem))
    num_images = len(polycam_image_filenames)

    # Load poses
    poses = []
    intrinsics = []
    for i, image_filename in enumerate(polycam_image_filenames):
        json_filename = Path(polycam_cameras_dir) / f"{image_filename.stem}.json"
        frame_json = load_from_json(json_filename)
        if i == 0:
            ori_width = frame_json["width"]
            ori_height = frame_json["height"]
        c2w = [
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
        ]
        intrinsic = get_intrinsic_matrices(frame_json['fx'], frame_json['fy'], frame_json['cx'], frame_json['cy'])
        poses.append(c2w)
        intrinsics.append(intrinsic)
    poses = np.stack(poses, axis=0).astype(np.float32)
    poses[...,:, 2] *= -1
    poses[..., :, 1] *= -1


    # Sample from original data
    assert len(polycam_depth_filenames) == num_images
    assert len(poses) == num_images
    idx = np.arange(num_images)
    if args.max_size != -1 and num_images > args.max_size:
        idx = np.round(np.linspace(0, num_images - 1, args.max_size)).astype(int)
    polycam_image_filenames = list(np.array(polycam_image_filenames)[idx])
    polycam_depth_filenames = list(np.array(polycam_depth_filenames)[idx])
    poses = to4x4(poses[idx])

    # Process pose
    for _id  in tqdm(range(len((poses)))):
        c2w = poses[_id]
        np.savetxt(f'{output_dir}/pose/{_id}.txt', c2w)

    # Copy images to output directory
    target_image_paths = []
    for _id in tqdm(range(len((polycam_image_filenames)))):
        image_path = polycam_image_filenames[_id]
        copied_image_path = target_image_dir / f"{_id}.jpg"
        target_image_paths.append(copied_image_path)

        color = cv2.imread(str(image_path))
        color = cv2.resize(color, (target_width, target_height))
        cv2.imwrite(str(copied_image_path), color)

    # Processs intrinsic
    intrinsic = np.array(intrinsics[0]).reshape((3, 3))

    intrinsic = scale_camera(intrinsic, (target_width, target_height), (ori_width, ori_height))
    np.savetxt(f'{output_dir}/intrinsic/intrinsic_color.txt', intrinsic)
    np.savetxt(f'{output_dir}/intrinsic/intrinsic_depth.txt', intrinsic)

    # Process depth
    target_depth_paths = []
    for _id in tqdm(range(len((polycam_depth_filenames)))):
        depth_path = polycam_depth_filenames[_id]
        target_depth_path = target_depth_dir / f"{_id}.png"

        target_depth_paths.append(target_depth_path)
        depth = cv2.imread(str(depth_path), -1)
        depth = cv2.resize(depth, (target_width, target_height))
        cv2.imwrite(str(target_depth_path), depth)

    with open(args.room_json, 'r') as f:
        data = json.load(f)
    output_dir = os.path.join(os.path.dirname(args.room_json), 'extracted')
    mesh_path = os.path.join(output_dir, 'fused_mesh.ply')
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)

    align_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    transform_matrix_t = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    alignment_matrix = np.eye(3)
    o3d_vertices = np.array(o3d_mesh.vertices)
    alignment_matrix = alignment_matrix.dot(np.linalg.inv(align_matrix)).dot(np.linalg.inv(transform_matrix_t)).dot([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).dot(roty(math.pi))
    np.savetxt(os.path.join(output_dir, 'alignment_matrix.txt'), alignment_matrix)
    o3d_vertices = o3d_vertices.dot(alignment_matrix)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(o3d_vertices)
    bboxes = [o3d_mesh]

    object_dict = {}
    for i, object in enumerate(data["objects"]):
        transform_matrix_t = np.array(object['transform']).reshape([4, 4]).T
        sizes = np.array(object['dimensions'])
        
        # Creat o3d oriented bounding box
        translate = transform_matrix_t[:3, 3]
        rotation = transform_matrix_t[:3, :3]
        
        bbox_o3d = o3d.geometry.OrientedBoundingBox(translate.reshape([3,1]), \
                                                    rotation, \
                                                    np.array(sizes).reshape([3,1]))
        bboxes.append(bbox_o3d)
        
        object_dict[i] = {'size': sizes, 'transform': transform_matrix_t, 'category':list(object['category'].keys())[0]}
    np.save(os.path.join(output_dir, 'objects.npy'), object_dict)