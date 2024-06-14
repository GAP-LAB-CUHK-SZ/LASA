import numpy as np
import os
import argparse
import open3d as o3d
import glob
import cv2
import copy

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

parser=argparse.ArgumentParser()
parser.add_argument("--data_folder",type=str,required=True)
parser.add_argument("--save_dir",type=str,default=r"../example_process_data")
parser.add_argument("--debug",action="store_true",default=False)
args=parser.parse_args()

print("processing %s"%(args.data_folder))

data_folder=args.data_folder
scene_name=os.path.basename(data_folder)
save_folder=os.path.join(args.save_dir,scene_name)
os.makedirs(save_folder,exist_ok=True)
color_folder=os.path.join(data_folder,"color")
depth_folder=os.path.join(data_folder,"depth")
pose_folder=os.path.join(data_folder,"pose")

print(color_folder)

color_list=glob.glob(color_folder+"/*.jpg")
image_id_list=[os.path.basename(item)[0:-4] for item in color_list]
image_id_list.sort()

bbox_path=os.path.join(data_folder,"objects.npy")
bboxes_dict=np.load(bbox_path,allow_pickle=True).item()

intrinsic_path=os.path.join(data_folder,"intrinsic","intrinsic_color.txt")
K=np.loadtxt(intrinsic_path)

align_path=os.path.join(data_folder,"alignment_matrix.txt")
align_matrix=np.loadtxt(align_path)
if align_matrix.shape[0]==3:
    new_align_matrix=np.eye(4)
    new_align_matrix[0:3,0:3]=align_matrix
    align_matrix=new_align_matrix

mesh_path=os.path.join(data_folder,"fused_mesh.ply")
o3d_mesh=o3d.io.read_triangle_mesh(mesh_path)
o3d_vertices = np.array(o3d_mesh.vertices)
o3d_vert_homo=np.concatenate([o3d_vertices,np.ones([o3d_vertices.shape[0],1])],axis=1)
align_o3d_vertices = np.dot(o3d_vert_homo,align_matrix)[:,0:3]
o3d_mesh.vertices = o3d.utility.Vector3dVector(align_o3d_vertices)
align_mesh_save_path=os.path.join(save_folder,"align_mesh.ply")
o3d.io.write_triangle_mesh(align_mesh_save_path,o3d_mesh)

x=np.linspace(-1,1,10)
y=np.linspace(-1,1,10)
z=np.linspace(-1,1,10)
X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
vox_coor=np.concatenate([X[:,:,:,np.newaxis],Y[:,:,:,np.newaxis],Z[:,:,:,np.newaxis]],axis=-1)
vox_coor=np.reshape(vox_coor,(-1,3))
#print(np.amin(vox_coor,axis=0),np.amax(vox_coor,axis=0))

pre_proj_mates={}
obj_points_dict={}
trans_mats={}
point_save_folder=os.path.join(save_folder,"5_partial_points")
os.makedirs(point_save_folder,exist_ok=True)
tran_save_folder=os.path.join(save_folder,"10_tran_matrix")
os.makedirs(tran_save_folder,exist_ok=True)
for object_id in bboxes_dict:
    object = bboxes_dict[object_id]
    category = object['category']
    sizes = object['size']
    sizes *= 1.1
    transform_matrix_t = np.array(object['transform']).reshape([4, 4])
    translate = transform_matrix_t[:3, 3]
    rotation = transform_matrix_t[:3, :3]

    bbox_o3d = o3d.geometry.OrientedBoundingBox(translate.reshape([3, 1]),
                                                rotation,
                                                np.array(sizes).reshape([3, 1]))
    crop_pcd = o3d_mesh.crop(bbox_o3d)
    crop_vert = np.asarray(crop_pcd.vertices)
    org_crop_vert = crop_vert[:, :]
    crop_vert = crop_vert - translate
    crop_vert = np.dot(crop_vert,np.linalg.inv(rotation).T)
    crop_vert[:, 2] *= -1
    bb_min, bb_max = np.amin(crop_vert, axis=0), np.amax(crop_vert, axis=0)
    max_length = (bb_max - bb_min).max()
    center = (bb_max + bb_min) / 2
    crop_vert = (crop_vert - center) / max_length * 2

    obj_points_dict[object_id]=crop_vert
    crop_pcd.vertices=o3d.utility.Vector3dVector(crop_vert)
    save_path=os.path.join(point_save_folder,category+"_%d.ply"%(object_id))
    o3d.io.write_triangle_mesh(save_path,crop_pcd)

    proj_mat = np.eye(4)
    scale_tran = np.eye(4)
    scale_tran[0, 0], scale_tran[1, 1], scale_tran[2, 2] = max_length / 2, max_length / 2, max_length / 2
    proj_mat = np.dot(proj_mat, scale_tran)
    center_tran = np.eye(4)
    center_tran[0:3, 3] = center
    proj_mat = np.dot(center_tran, proj_mat)
    invert_mat = np.eye(4)
    invert_mat[2, 2] *= -1
    proj_mat = np.dot(invert_mat, proj_mat)
    proj_mat[0:3, 0:3] = np.dot(rotation,proj_mat[0:3, 0:3])
    translate_mat = np.eye(4)
    translate_mat[0:3, 3] = translate
    proj_mat = np.dot(translate_mat, proj_mat)

    '''tran mat is to align output to scene space'''
    tran_mat=copy.deepcopy(proj_mat)
    trans_mats[object_id]=tran_mat
    tran_save_path=os.path.join(tran_save_folder,category+"_%d.npy"%(object_id))
    np.save(tran_save_path,tran_mat)

    unalign_mat = np.linalg.inv(align_matrix)
    proj_mat = np.dot(unalign_mat.T, proj_mat)
    pre_proj_mates[object_id]=proj_mat

ref=np.array([
            [0,1.0], #Up
            [-1.0,0],#Left
            [0,1.0], #Right
            [0.0,-1.0] #Down
        ]) #4*2
dir_list=[
    "Down",
    "Left",
    "Right",
    "Up"
]

for image_id in image_id_list:
    color_path=os.path.join(color_folder,image_id+".jpg")
    depth_path=os.path.join(depth_folder,image_id+".png")
    pose_path=os.path.join(pose_folder,image_id+".txt")

    color=cv2.imread(color_path)
    height,width=color.shape[0:2]
    depth=cv2.imread(depth_path,cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)/1000.0
    pose=np.loadtxt(pose_path)
    for object_id in bboxes_dict:
        object=bboxes_dict[object_id]
        category=object['category']
        sizes=object['size']
        object_vox_coor=vox_coor*sizes[np.newaxis,:]
        #print(np.amin(object_vox_coor,axis=0),np.amax(object_vox_coor,axis=0))
        #print(sizes)

        prev_proj_mat=pre_proj_mates[object_id]
        wrd2cam_pose = np.linalg.inv(pose)
        current_proj_mat = np.dot(wrd2cam_pose, prev_proj_mat)
        K4=np.eye(4)
        K4[0:3,0:3]=K

        '''calibrate proj_mat'''
        up_vectors = np.array([[0, 0, 0, 1.0],
                               [0, 0.5, 0, 1.0]])
        up_vec_inimg = np.dot(up_vectors, current_proj_mat.T)
        up_vec_inimg = np.dot(up_vec_inimg,K4.T)
        up_x = up_vec_inimg[:, 0] / up_vec_inimg[:, 2]
        up_y = up_vec_inimg[:, 1] / up_vec_inimg[:, 2]
        pt1 = np.array((up_x[0], up_y[0]))
        pt2 = np.array((up_x[1], up_y[1]))
        up_dir = pt2 - pt1
        # print(up_dir)

        product = np.sum(up_dir[np.newaxis, :] * ref, axis=1)
        max_ind = np.argmax(product)
        direction = dir_list[max_ind]
        sky_rot = rotate_mat(direction)
        #final_proj_mat = np.dot(K4,final_proj_mat)

        vox_homo=np.concatenate([object_vox_coor,np.ones((object_vox_coor.shape[0],1))],axis=1)
        vox_proj=np.dot(vox_homo,current_proj_mat.T)
        vox_proj=np.dot(vox_proj,K4.T)
        vox_x=vox_proj[:,0]/vox_proj[:,2]
        vox_y=vox_proj[:,1]/vox_proj[:,2]

        if np.mean(vox_proj[:,2])>5:
            continue

        inside_mask=((vox_x<width-1) &(vox_x>0) &(vox_y<height-1) &(vox_y>0)).astype(np.float32)
        infrustum_ratio=np.sum(inside_mask)/vox_x.shape[0]
        if infrustum_ratio < 0.4 and category in ["chair", "stool"]:
            continue
        elif infrustum_ratio <0.2:
            continue
        #print(object_id,image_id,infrustum_ratio)

        '''objects visibility check for every frame'''
        vox_x_inside=vox_x[inside_mask>0].astype(np.int32)
        vox_y_inside=vox_y[inside_mask>0].astype(np.int32)
        vox_depth=vox_proj[inside_mask>0,2]
        #print(depth.shape,np.amax(vox_y_inside),np.amax(vox_x_inside))
        depth_sample=depth[vox_y_inside,vox_x_inside]
        depth_mask=(depth_sample>0)&(depth_sample<10.0)
        depth_sample=depth_sample[depth_mask]
        vox_depth=vox_depth[depth_mask]

        if vox_depth.shape[0]<100:
            continue

        occluded_ratio=np.sum(((vox_depth-depth_sample)>0.2).astype(np.float32))/vox_depth.shape[0]
        if occluded_ratio>0.6 and category in ["chair"]: #chair is easily occluded, while table is not
            continue

        depth_near_ratio = np.sum((np.abs(vox_depth - depth_sample) < sizes.max() * 0.5).astype(np.float32)) / \
                           vox_depth.shape[0]
        if depth_near_ratio < 0.2:
            continue

        '''make sure in every image, the object is upward'''
        bbox=(np.amin(vox_x_inside),np.amin(vox_y_inside),np.amax(vox_x_inside),np.amax(vox_y_inside))
        rot_image=rotate_image(color,direction)
        bbox = rotate_bbox(bbox, direction, height, width)
        crop_image=rot_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        crop_h, crop_w = crop_image.shape[0:2]
        max_length = max(crop_h, crop_w)
        if max_length<100:
            continue
        pad_image = np.zeros((max_length, max_length, 3))
        if crop_h > crop_w:
            margin = crop_h - crop_w
            pad_image[:, margin // 2:margin // 2 + crop_w] = crop_image[:, :, :]
            x_start, x_end = bbox[0] - margin // 2, margin // 2 + bbox[2]
            y_start, y_end = bbox[1], bbox[3]
        else:
            margin = crop_w - crop_h
            pad_image[margin // 2:margin // 2 + crop_h, :] = crop_image[:, :, :]

            y_start, y_end = bbox[1] - margin // 2, bbox[3] + margin // 2
            x_start, x_end = bbox[0], bbox[2]

        pad_image=cv2.resize(pad_image,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
        image_save_folder = os.path.join(save_folder, "6_images", category + "_%d" % (object_id))
        os.makedirs(image_save_folder, exist_ok=True)
        image_save_path=os.path.join(image_save_folder,image_id+".jpg")
        #print("saving to %s"%(image_save_path))
        cv2.imwrite(image_save_path,pad_image)

        proj_mat=np.dot(sky_rot,current_proj_mat)
        new_K4 = rotate_K(K, direction)
        new_K4[0, 2] -= x_start
        new_K4[1, 2] -= y_start
        new_K4[0] = new_K4[0] / max_length * 224
        new_K4[1] = new_K4[1] / max_length * 224
        proj_mat = np.dot(new_K4, proj_mat)

        proj_save_folder=os.path.join(save_folder,"8_proj_matrix",category+"_%d"%(object_id))
        os.makedirs(proj_save_folder,exist_ok=True)
        proj_save_path=os.path.join(proj_save_folder,image_id+".npy")
        np.save(proj_save_path,proj_mat)

        '''debug proj matrix'''
        if args.debug:
            proj_save_folder=os.path.join(save_folder,"9_proj_images",category+"_%d"%(object_id))
            os.makedirs(proj_save_folder,exist_ok=True)
            canvas=copy.deepcopy(pad_image)
            par_points=obj_points_dict[object_id]
            par_homo=np.concatenate([par_points,np.ones((par_points.shape[0],1))],axis=1)
            par_inimg=np.dot(par_homo,proj_mat.T)
            x=par_inimg[:,0]/par_inimg[:,2]
            y=par_inimg[:,1]/par_inimg[:,2]
            x=np.clip(x,a_min=0,a_max=223).astype(np.int32)
            y=np.clip(y,a_min=0,a_max=223).astype(np.int32)
            canvas[y,x]=np.array([[0,255,0]])
            proj_save_path=os.path.join(proj_save_folder,image_id+".jpg")
            cv2.imwrite(proj_save_path,canvas)



