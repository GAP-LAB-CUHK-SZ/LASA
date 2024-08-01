import numpy as np
import scipy
import os
import trimesh
from sklearn.cluster import KMeans
import random
import glob
import tqdm
import argparse
import multiprocessing as mp
import sys
sys.path.append("..")

parser=argparse.ArgumentParser()
parser.add_argument("--keyword",type=str,default="lowres") #augment only the low resolution points
parser.add_argument("--data_root",type=str,default="../data/other_data")
args=parser.parse_args()
category_list=os.listdir(args.data_root)
category_list=[category for category in category_list if "arkit_" in category]
kmeans=KMeans(
    init="random",
    n_clusters=20,
    n_init=10,
    max_iter=300,
    random_state=42
)

def process_data(src_point_path,save_folder,keyword):
    src_point_tri = trimesh.load(src_point_path)
    src_point = np.asarray(src_point_tri.vertices)
    kmeans.fit(src_point)
    point_cluster_index = kmeans.labels_

    '''choose 10~19 clusters to form the augmented new point'''
    for i in range(10):
        n_cluster = random.randint(14, 19)  # 14,19 for lowres, 10,19 for highres
        choose_cluster = np.random.choice(20, n_cluster, replace=False)
        aug_point_list = []
        for cluster_index in choose_cluster:
            cluster_point = src_point[point_cluster_index == cluster_index]
            aug_point_list.append(cluster_point)
        aug_point = np.concatenate(aug_point_list, axis=0)
        save_path = os.path.join(save_folder, "%s_partial_points_%d.ply" % (keyword, i + 1))
        print("saving to %s"%(save_path))
        aug_point_tri = trimesh.PointCloud(vertices=aug_point)
        aug_point_tri.export(save_path)

pool=mp.Pool(10)
for cat in category_list[0:]:
    keyword=args.keyword
    point_dir = os.path.join(args.data_root,cat,"5_partial_points")
    folder_list=os.listdir(point_dir)
    for folder in tqdm.tqdm(folder_list[0:]):
        folder_path=os.path.join(point_dir,folder)
        src_point_path=os.path.join(point_dir,folder,"%s_partial_points_0.ply"%(keyword))
        if os.path.exists(src_point_path)==False:
            continue
        save_folder=folder_path
        pool.apply_async(process_data,(src_point_path,save_folder,keyword))
pool.close()
pool.join()