import os
import mesh_to_sdf
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["OMP_NUM_THREADS"]="2"
import trimesh
import numpy as np
import glob
import multiprocessing as mp
import argparse
def process_object(folder,occ_save_dir,other_save_dir,consider_alignment):
    object_id = folder.split(os.sep)[-1]
    scene_id = folder.split(os.sep)[-3]
    mesh_path = os.path.join(folder,"%s_watertight.obj"%(object_id))
    print("processing %s"%(mesh_path))
    highres_partial_path=os.path.join(folder,"%s_laser_pcd.ply"%(object_id))
    lowres_partial_path=os.path.join(folder,"%s_rgbd_mesh.ply"%(object_id))

    if os.path.exists(mesh_path)==False or os.path.exists(highres_partial_path)==False or \
        os.path.exists(lowres_partial_path)==False:
        return

    if consider_alignment:
        align_mat_path = os.path.join(folder, "alignment.txt")
        align_mat=np.loadtxt(align_mat_path)
    else:
        align_mat=np.eye(4)
    occ_save_path=os.path.join(occ_save_dir,object_id+".npz")
    scale_save_path=os.path.join(occ_save_dir,object_id+".npy")
    if os.path.exists(occ_save_path):
       print("skipping %s" % (occ_save_path))
       return
    tri_mesh=trimesh.load(mesh_path)
    highres_partial_point=trimesh.load(highres_partial_path)
    highres_partial_vert=np.asarray(highres_partial_point.vertices)

    lowres_partial_point = trimesh.load(lowres_partial_path)
    lowres_partial_vert = np.asarray(lowres_partial_point.vertices)

    '''normalize '''
    vert = np.asarray(tri_mesh.vertices)
    vert=np.dot(vert,align_mat[0:3,0:3].T)+align_mat[0:3,3] # align with the rgb-d points
    vert=vert[:,[1,2,0]] #align with shapenet coordinate
    vert[:,2]*=-1 #align with shapenet coordinate
    x_min, x_max = np.amin(vert[:, 0]), np.amax(vert[:, 0])
    y_min, y_max = np.amin(vert[:, 1]), np.amax(vert[:, 1])
    z_min, z_max = np.amin(vert[:, 2]), np.amax(vert[:, 2])
    bbox_size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_length = np.amax(bbox_size)
    center=np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2])
    vert=(vert-center)/max_length*2

    #tran_mat is used to put the results back to the scene
    tran_mat=np.eye(4)
    tran_mat=np.dot(align_mat,tran_mat)
    tran_mat=np.dot(np.array([[0,1,0,0],
                              [0,0,1,0],
                              [-1,0,0,0],
                              [0,0,0,1]]),tran_mat)
    center_mat=np.eye(4)
    center_mat[0:3,3]=-center
    tran_mat=np.dot(center_mat,tran_mat)
    tran_mat=np.dot(np.eye(4)/max_length*2,tran_mat)
    tran_mat_save_folder=os.path.join(other_save_dir,"10_tranmat")
    os.makedirs(tran_mat_save_folder,exist_ok=True)
    np.save(os.path.join(tran_mat_save_folder,"tranmat.npy"),{"scene_id":scene_id,"tranmat":tran_mat},allow_pickle=True)

    highres_partial_vert=np.dot(highres_partial_vert[:,0:3],align_mat[0:3,0:3].T)+align_mat[0:3,3] #align laser points to rgbd points
    highres_partial_vert=highres_partial_vert[:,[1,2,0]]
    highres_partial_vert[:,2]*=-1

    lowres_partial_vert=lowres_partial_vert[:,[1,2,0]]
    lowres_partial_vert[:,2]*=-1
    highres_partial_vert=(highres_partial_vert-center)/max_length*2
    lowres_partial_vert=(lowres_partial_vert-center)/max_length*2

    tri_mesh.vertices=vert[:,:].copy()
    lowres_partial_point.vertices=lowres_partial_vert[:,:].copy()
    highres_partial_point.vertices=highres_partial_vert[:,:].copy()
    surface_point=trimesh.sample.sample_surface(tri_mesh,231000)[0]


    nss_samples=trimesh.sample.sample_surface(tri_mesh,365000)[0]
    nss_samples=nss_samples+np.random.randn(nss_samples.shape[0],3)*max_length*0.07

    vol_points = (np.random.random((64 * 64 * 64, 3)) - 0.5) * 2

    nss_sdf=mesh_to_sdf.mesh_to_sdf(tri_mesh,nss_samples)
    grid_sdf=mesh_to_sdf.mesh_to_sdf(tri_mesh,vol_points)
    grid_label=(grid_sdf<0).astype(bool)
    near_label=(nss_sdf<0).astype(bool)
    os.makedirs(occ_save_dir,exist_ok=True)
    print("saving to %s"%(occ_save_path))
    np.savez_compressed(occ_save_path,vol_points=vol_points.astype(np.float32),
                        vol_label=grid_label,near_points=nss_samples.astype(np.float32),
                        near_label=near_label)
    point_cloud_savefolder=os.path.join(other_save_dir,"4_pointcloud")
    if os.path.exists(point_cloud_savefolder)==False:
        os.makedirs(point_cloud_savefolder)
    point_cloud_savepath=os.path.join(point_cloud_savefolder,object_id+".npz")
    print("saving to %s"%(point_cloud_savepath))
    np.savez_compressed(point_cloud_savepath,points=surface_point.astype(np.float32))

    scale = np.array([1.0])
    np.save(scale_save_path, scale)

    par_point_savefolder=os.path.join(other_save_dir,"5_partial_points",object_id)
    if os.path.exists(par_point_savefolder)==False:
        os.makedirs(par_point_savefolder)
    highres_savepath=os.path.join(par_point_savefolder,"highres_partial_points_0.ply")
    lowres_savepath = os.path.join(par_point_savefolder, "lowres_partial_points_0.ply")
    print("saving to %s"%(highres_savepath))
    highres_partial_point.export(highres_savepath)
    lowres_partial_point.export(lowres_savepath)


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--lasa_dir",type=str,required=True)
parser.add_argument("--save_dir",type=str,default="../data")
parser.add_argument("--consider_alignment",action="store_true")
args=parser.parse_args()
lasa_dir=args.lasa_dir
save_dir=args.save_dir
consider_alignment=args.consider_alignment

if __name__=="__main__":
    pool = mp.Pool(1)
    scene_id_list=os.listdir(lasa_dir)
    for scene_id in scene_id_list[0:10]:
        scene_folder=os.path.join(lasa_dir,scene_id)
        instance_folder=os.path.join(scene_folder,"instances")
        if os.path.exists(instance_folder)==False:
            continue
        object_id_list=os.listdir(instance_folder)
        for object_id in object_id_list:
            object_folder=os.path.join(instance_folder,object_id)
            category="_".join(object_id.split("_")[0:-1])
            occ_save_dir=os.path.join(save_dir,"occ_data","arkit_"+category)
            other_save_dir=os.path.join(save_dir,"other_data","arkit_"+category,object_id)
            pool.apply_async(process_object,(object_folder,occ_save_dir,other_save_dir,consider_alignment))
            #process_object(object_folder,occ_save_dir,other_save_dir,consider_alignment)
        #process_object(folder_path)
    pool.close()
    pool.join()