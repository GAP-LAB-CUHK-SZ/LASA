## Raw Dataset processing for reconstruction training
The procedure includes several steps. It firstly **select some frames** from the raw data, in which the
objects are visible. Then, **super resolution** is employed to upscale the original low resolution images.
Finally, it will **crop the objects in the image**, and **recompute the camera pose and 
intrinsic matrix** for each image. All the scripts are inside <a href=https://github.com/GAP-LAB-CUHK-SZ/LASA/arkitscene_process_script>./arkitscene_process_script</a>.
<br>
Firstly, download the ArkitScene 3dod dataset from <a href="https://github.com/apple/ARKitScenes">ArkitScene's repository</a>. We provide a list of annotated scene in
<a href=https://github.com/GAP-LAB-CUHK-SZ/LASA/arkitscene_process_script/annotate_scene_list.txt>annotate_scene_list.txt</a>. Only these scene need to be downloaded.
You can put the <a href="https://github.com/apple/ARKitScenes/blob/main/download_data.py">download_data.py</a> under LASA/arkitscene_process_script and use the following script to download the selected data:
```angular2html
python donwload_select_arkitscene.py --save_dir <path_to_arkit>
```

Unzip all files by the following commands.
```angular2html
python unzip_arkit_data.py --arkit_dir <path_to_arkit_dataset/3dod> --split Training
python unzip_arkit_data.py --arkit_dir <path_to_arkit_dataset/3dod> --split Validation
```
Secondly, select some images such that the object are visible in this images by the following command:
```angular2html
python select_arkitscene_images.py --arkit_root <path_to_arkit_dataset/3dod> --save_root <path_to_arkit_dataset/images> --split Training
python select_arkitscene_images.py --arkit_root <path_to_arkit_dataset/3dod> --save_root <path_to_arkit_dataset/images> --split Validation
```
Thirdly, install <a href="https://github.com/XPixelGroup/HAT"> HAT <a/> for super resolution on the low resolution images.
Download the pretrained model HAT_SRx4_ImageNet-pretrain.pth from them, and put the file under /LASA/checkpoint/SR_model/HAT_SRx4_ImageNet-pretrain.pth. 
Then, run the following command to super resolution the images.
```angular2html
CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15000 --nproc_per_node=4 \
SR_images.py --image_dir <path_to_arkit_dataset/images>
```
Then, run the following command to crop and pad the images, meanwhile recompute the projection matrix, so
that the points can be projected to the crop images directly using this matrix:
```angular2html
python crop_arkit_images.py --image_dir <path_to_arkit_dataset/images> --lasa_dir <path_to_LASA_dataset> \
--arkit_dir <path_to_arkit_dataset/3dod> --consider_alignment
```
The mesh annotation is originally aligned with the LiDAR scan, and might be slightly misaligned with the RGB-D scan.
Therefore, we set --consider_alignment flag, which is used to further align the annotation with the RGB-D scan. 
The <object_id>_gt_mesh_2.ply will be better aligned with the RGB-D scan if you multiply it with the alignment matrix.
<br>

Next, Install <a href="https://github.com/hjwdzh/ManifoldPlus">ManifoldPlus</a>, and add the **build** folder to
the **PATH** environment variable. Then, convert the gt mesh into watertight mesh by the following commands:
```angular2html
python convert_watertight.py --lasa_dir <path_to_LASA_dataset>
```


Next, generate occupancy GT for the annotation, and format the RGB-D and LiDAR point cloud for inputs.
```angular2html

```
In case of running on a headless machine, you can refer to 
<a href="https://pyrender.readthedocs.io/en/latest/install/index.html?highlight=ssh#getting-pyrender-working-with-osmesa">this page</a> for
how to use Mesa for computing the sdf or occupancy values.

## Download preprocessed data and processing
You can choose to process the raw data of LASA by yourself, or download the preprocessed data from <a href="https://pan.baidu.com/s/1X6k82UNG-1hV_FIthnlwcQ?pwd=r7vs">
BaiduYun (code: r7vs)<a/> or <a href="https://cuhko365.sharepoint.com/:f:/s/CUHKSZ_SSE_GAP-Lab2/EmMw149zXuhNuWzJMVxvF7kBfUEKUkKpYO6apJNw0HSKqA?e=hEMRUh">Onedrive SharePoint<a/>. 
Put all the downloaded data under LASA, unzip the align_mat_all.zip mannually. 
Currently, the synthetic dataset such as ShapeNet, ABO, and 3D-FUTURE only provide preprocessed data for download. 
Then, use the script LASA/process_scripts/unzip_all_data to unzip all the data in occ_data and other_data by following commands:
```angular2html
cd process_scripts
python unzip_all_data.py --unzip_occ --unzip_other
```
Run the following commands to generate augmented partial point cloud for synthetic dataset and LASA dataset
```angular2html
cd process_scripts
python augment_arkit_partial_point.py --cat arkit_chair arkit_stool ...
python augment_synthetic_partial_point.py --cat 03001627 future_chair ABO_chair future_stool ...
```
Run the following command to extract image features
```angular2html
cd process_scripts
bash dist_extract_vit.sh
```
Finally, run the following command to generate train/val splits, please check ./dataset/taxonomy for the sub-cateory definition:
```angular2html
cd process_scripts
python generate_split_for_arkit --cat arkit_chair arkit_stool ...
python generate_split_for_synthetic_data.py --cat 03001627 future_chair ABO_chair future_stool ...