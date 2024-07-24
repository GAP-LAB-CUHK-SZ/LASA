import copy

import numpy as np
import os
import glob
import cv2
import trimesh
import tqdm

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

def rotate_bbox(bbox,direction, H,W):

    x_min,y_min,x_max,y_max=bbox[0:4]
    if direction == 'Up':
        return bbox
    elif direction == 'Left':
        #print(W-bbox[1],W-bbox[3])
        new_bbox=[min(H-bbox[1],H-bbox[3]),bbox[0],max(H-bbox[1],H-bbox[3]),bbox[2]]
    elif direction == 'Right':
        new_bbox=[bbox[1],min(W-bbox[0],W-bbox[2]),bbox[3],max(W-bbox[0],W-bbox[2])]
    elif direction == 'Down':
        new_bbox=[min(W-x_min,W-x_max),min(H-y_min,H-y_max),max(W-x_min,W-x_max),max(H-y_min,H-y_max)]
    else:
        raise Exception(f'No such direction (={direction}) rotation')
    return new_bbox

def get_roll_rot(angle):
    ca=np.cos(angle)
    sa=np.sin(angle)
    rot=np.array([
        [ca,-sa,0,0],
        [sa,ca,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return rot

def rotate_mat(direction):
    if direction == 'Up':
        return np.eye(4)
    elif direction == 'Left':
        rot_mat=get_roll_rot(np.pi/2)
    elif direction == 'Right':
        rot_mat=get_roll_rot(-np.pi/2)
    elif direction == 'Down':
        rot_mat=get_roll_rot(np.pi)
    else:
        raise Exception(f'No such direction (={direction}) rotation')
    return rot_mat

def rotate_K(K,direction):
    if direction == 'Up' or direction=="Down":
        new_K4=np.eye(4)
        new_K4[0:3,0:3]=copy.deepcopy(K)
        return new_K4
    elif direction == 'Left' or direction =="Right":
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        new_K4 = np.array([
            [fy, 0, cy, 0],
            [0, fx, cx, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return new_K4

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--save_dir",default="../data",type=str)
parser.add_argument("--image_dir",required=True,type=str)
parser.add_argument("--lasa_dir",required=True,type=str)
parser.add_argument("--arkit_dir",required=True,type=str)
parser.add_argument("--consider_alignment",action="store_true") #the annotation is originally aligned with LiDAR scan,
# using alignment matrix, the annotation will be more aligned to the RGB-D scan

args=parser.parse_args()
other_save_dir=os.path.join(args.save_dir,"other_data")
image_dir=args.image_dir
lasa_dir=args.lasa_dir
arkit_dir=args.arkit_dir

scene_id_list=os.listdir(image_dir)
for scene_id in scene_id_list:
    object_id_list=os.listdir(os.path.join(image_dir,scene_id))
    for object_id in object_id_list:
        print("processing %s %s"%(scene_id,object_id))
        sub_category = "_".join(object_id.split("_")[0:-1])
        #print(sub_category)

        mesh_folder=os.path.join(lasa_dir,scene_id,"instances",object_id)
        mesh_path=glob.glob(mesh_folder+"/%s_gt*.obj"%(object_id))
        #print(mesh_path)
        if len(mesh_path)==0: #this instance does not have annotation
            continue
        mesh_path=mesh_path[0]
        mesh=trimesh.load(mesh_path)

        if args.consider_alignment:
            align_mat_path=os.path.join(mesh_folder,"alignment.txt")
            align_mat=np.loadtxt(align_mat_path)

        vert = np.asarray(mesh.vertices)
        if args.consider_alignment: vert = np.dot(vert, align_mat[0:3, 0:3].T) + align_mat[0:3, 3]
        vert = vert[:, [1, 2, 0]]
        vert[:, 2] *= -1 #all coordinate are aligned with shapenet dataset
        bb_min = np.amin(vert, axis=0)
        bb_max = np.amax(vert, axis=0)
        center = (bb_min + bb_max) / 2
        size = (bb_max - bb_min)
        max_length = size.max()
        up_vector = np.array([[0, -1.0, 0],
                              [0, 1.0, 0]])  # 2x3

        image_folder = os.path.join(image_dir, scene_id, object_id,"highres_color")
        lowres_image_folder = os.path.join(image_dir, scene_id, object_id,"color")
        direction_path = os.path.join(image_dir,scene_id,object_id,"direction.txt")
        proj_mat_folder = os.path.join(image_dir, scene_id, object_id, "proj_matrix")
        image_save_folder=os.path.join(other_save_dir,"arkit_"+sub_category,"6_images",object_id)
        proj_mat_save_folder=os.path.join(other_save_dir,"arkit_"+sub_category,"8_proj_matrix",object_id)

        image_list = os.listdir(image_folder)

        intrinsic_folder = os.path.join(arkit_dir,"Training", scene_id, scene_id+"_frames","lowres_wide_intrinsics")
        if os.path.exists(intrinsic_folder)==False:
            intrinsic_folder = os.path.join(arkit_dir, "Validation", scene_id, scene_id + "_frames",
                                            "lowres_wide_intrinsics")
        intrinsic_path=os.path.join(intrinsic_folder,os.listdir(intrinsic_folder)[0])
        K = st2_camera_intrinsics(intrinsic_path)

        old_K = copy.deepcopy(K[:, :])
        old_K4 = np.eye(4)
        old_K4[0:3, 0:3] = old_K
        K[0] = K[0] * 4
        K[1] = K[1] * 4  # image is super-resoluted

        '''recompute calibration matrix'''
        scale_mat = np.eye(4)
        scale_mat[0, 0], scale_mat[1, 1], scale_mat[2, 2] = max_length / 2, max_length / 2, max_length / 2
        center_mat = np.eye(4)
        center_mat[0:3, 3] = center
        z_invert = np.eye(4)
        z_invert[2, 2] = -1.0
        permute_mat = np.eye(4)
        permute_mat = permute_mat[[2, 0, 1, 3]]
        if args.consider_alignment:
            align_inv = np.linalg.inv(align_mat)

        calib = np.dot(center_mat, scale_mat)
        calib = np.dot(z_invert, calib)
        calib = np.dot(permute_mat, calib)
        if args.consider_alignment:
            calib = np.dot(align_inv, calib)

        ref = np.array([
            [0, 1.0],  # Up
            [-1.0, 0],  # Left
            [1.0, 0.0],  # Right
            [0.0, -1.0]  # Down
        ])  # 4*2
        dir_list = [
            "Down",
            "Left",
            "Right",
            "Up"
        ]
        for image_filename in image_list:
            image_id = image_filename[:-4]
            image_path = os.path.join(image_folder, image_filename)
            # print(image_path)
            image = cv2.imread(image_path)
            height, width = image.shape[0:2]
            bbox2d_path = os.path.join(lowres_image_folder, image_filename[:-4] + ".npy")
            bbox2d = np.load(bbox2d_path)

            proj_mat_path = os.path.join(proj_mat_folder, image_filename[:-4] + ".npy")
            org_proj_mat = np.load(proj_mat_path)
            '''calibrate proj_mat'''
            proj_mat = np.dot(org_proj_mat, calib)

            '''rotate the image, so that the objects is always upward'''
            up_vectors = np.array([[0, 0, 0, 1.0],
                                   [0, 0.5, 0, 1.0]])
            up_vec_inimg = np.dot(up_vectors, proj_mat.T)
            up_x = up_vec_inimg[:, 0] / up_vec_inimg[:, 2]
            up_y = up_vec_inimg[:, 1] / up_vec_inimg[:, 2]
            pt1 = np.array((up_x[0], up_y[0]))
            pt2 = np.array((up_x[1], up_y[1]))
            up_dir = pt2 - pt1

            product = np.sum(up_dir[np.newaxis, :] * ref, axis=1)
            max_ind = np.argmax(product)
            direction = dir_list[max_ind]
            sky_rot = rotate_mat(direction)

            # print("org bbox",bbox)
            bbox2d = bbox2d * 4
            bbox2d[0] = max(0, bbox2d[0])
            bbox2d[2] = min(width - 1, bbox2d[2])
            bbox2d[1] = max(0, bbox2d[1])
            bbox2d[3] = min(height - 1, bbox2d[3])
            # print("large bbox",bbox)
            x_size = bbox2d[2] - bbox2d[0]
            y_size = bbox2d[3] - bbox2d[1]
            if max(x_size, y_size) < 256:
                continue
            if x_size / y_size > 2:
                continue
            if y_size / x_size > 2:
                continue

            image = rotate_image(image, direction)
            bbox2d = rotate_bbox(bbox2d, direction, height, width)
            crop_image = image[bbox2d[1]:bbox2d[3], bbox2d[0]:bbox2d[2], :]

            crop_h, crop_w = crop_image.shape[0:2]
            max_length = max(crop_h, crop_w)
            pad_image = np.zeros((max_length, max_length, 3))
            if crop_h > crop_w:
                margin = crop_h - crop_w
                pad_image[:, margin // 2:margin // 2 + crop_w] = crop_image[:, :, :]
                x_start, x_end = bbox2d[0] - margin // 2, margin // 2 + bbox2d[2]
                y_start, y_end = bbox2d[1], bbox2d[3]
            else:
                margin = crop_w - crop_h
                pad_image[margin // 2:margin // 2 + crop_h, :] = crop_image[:, :, :]

                y_start, y_end = bbox2d[1] - margin // 2, bbox2d[3] + margin // 2
                x_start, x_end = bbox2d[0], bbox2d[2]
            img_save_path = os.path.join(image_save_folder, image_filename)
            os.makedirs(image_save_folder,exist_ok=True)
            cv2.imwrite(img_save_path, pad_image)

            '''then recompute the projection matrix on crop image'''
            proj_mat_wo_K = np.dot(np.linalg.inv(old_K4), proj_mat)
            proj_mat_wo_K = np.dot(sky_rot, proj_mat_wo_K)
            new_K4 = rotate_K(K, direction)

            new_K4[0, 2] -= x_start
            new_K4[1, 2] -= y_start
            new_K4[0] = new_K4[0] / max_length * 224
            new_K4[1] = new_K4[1] / max_length * 224
            new_proj_mat = np.dot(new_K4, proj_mat_wo_K)

            proj_save_path = os.path.join(proj_mat_save_folder, image_filename[:-4] + ".npy")
            os.makedirs(proj_mat_save_folder,exist_ok=True)
            np.save(proj_save_path, new_proj_mat)
