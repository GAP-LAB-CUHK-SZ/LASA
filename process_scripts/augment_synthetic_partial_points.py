import numpy as np
import scipy
import os
import trimesh
from sklearn.cluster import KMeans
import random
import glob
import tqdm
import multiprocessing as mp
import sys
sys.path.append("..")
from datasets.taxonomy import synthetic_category_combined

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--root_dir",type=str,default="../data/other_data")
args=parser.parse_args()
category_list=os.listdir(args.data_root)
category_list=[category for category in category_list if "arkit_" not in category]

kmeans=KMeans(
    init="random",
    n_clusters=7,
    n_init=10,
    max_iter=300,
    random_state=42
)

def process_data(src_filepath,save_path):
    #print("processing %s"%(src_filepath))
    src_point_tri = trimesh.load(src_filepath)
    src_point = np.asarray(src_point_tri.vertices)
    kmeans.fit(src_point)
    point_cluster_index = kmeans.labels_

    n_cluster = random.randint(3, 6)
    choose_cluster = np.random.choice(7, n_cluster, replace=False)
    aug_point_list = []
    for cluster_index in choose_cluster:
        cluster_point = src_point[point_cluster_index == cluster_index]
        aug_point_list.append(cluster_point)
    aug_point = np.concatenate(aug_point_list, axis=0)
    aug_point_tri = trimesh.PointCloud(vertices=aug_point)
    print("saving to %s"%(save_path))
    aug_point_tri.export(save_path)

pool=mp.Pool(10)
for cat in category_list:
    print("processing %s"%cat)
    point_dir=os.path.join(args.root_dir,cat,"5_partial_points")
    folder_list=os.listdir(point_dir)
    for folder in folder_list[:]:
        folder_path=os.path.join(point_dir,folder)
        src_filelist=glob.glob(folder_path+"/partial_points_*.ply")
        for src_filepath in src_filelist:
            basename=os.path.basename(src_filepath)
            save_path = os.path.join(point_dir, folder, "aug7_" + basename)
            pool.apply_async(process_data,(src_filepath,save_path))
pool.close()
pool.join()