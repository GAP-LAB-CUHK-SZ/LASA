## Download preprocessed data and processing (Already released)
You can choose to process the raw data of LASA by yourself, or download the preprocessed data from <a href="https://cuhko365.sharepoint.com/:f:/s/CUHKSZ_SSE_GAP-Lab2/Eu_V8OWr7VdDj--6xzWm2AwBuJsDBOLHA2z09D29cQtq_g?e=fkEpPo">Onedrive SharePoint<a/>. 

Format of the prepare data is as follows:
```
other_data/
└── category/
    ├── 4_pointcloud     # point clouds sampled from CAD annotation
    ├── 6_images         # images that contains this images
    ├── 8_proj_matrix    # project matrix, that project 3d points to 2D image coordinate
    └── 5_partial_points/        # partial point clouds cropped from laser scan and rgb-d scan
        └── cadid/
                ├── highres_partial_points_0.ply    # point clouds cropped from laser scan
                └── lowres_partial_points_0.ply    	# point clouds cropped from rgb-d scan
 
occ_data/
└── category/
        ├── cadid.npz   # contains samples and occupancy data 
        └── cadid.npy   # contains scale information (default is 1, except for ShapeNet dataset)
```
The category and sub-category mapping is as followed:

| Category | 3D-FUTURE                  | ABO          | ShapeNet   | LASA                                                            |
|----------|----------------------------|--------------|------------|-----------------------------------------------------------------|
| chair    | future_chair, future_stool | ABO_chair    | 03001627   | arkit_chair, arkit_stool                                        |  
| table    | future_table               | ABO_table    | 04379243   | arkit_table                                                     |
| sofa     | future_sofa                | ABO_sofa     | 04256520   | arkit_sofa                                                      |
| bed      | future_bed                 | ABO_bed      | 02818832   | arkit_bed                                                       |
| shelf    | future_shelf               | ABO_shelf    | 02871439   | arkit_shelf                                                     |
| cabinet  | future_cabinet             | ABO_cabinet  | 02933112, 03207941 | arkit_cabinet, arkit_dishwasher, arkit_refrigerator, arkit_oven |

Put **occ_data** **other_data** under **/LASA/submodules/DisCo/data**, put **open_clip_pytorch_model.bin** under **/LASA/submodules/DisCo/data**. 
Currently, the synthetic dataset such as ShapeNet, ABO, and 3D-FUTURE only provide preprocessed data for download. 
Then, run the following command to **unzip all the data** in **occ_data** and **other_data** by following commands (PS. make sure unrar and 7zip is installed):
```angular2html
cd LASA/process_scripts
python unzip_all_data.py --unzip_occ --unzip_other
```
Run the following commands to generate **augmented partial point cloud** for synthetic dataset and LASA dataset
```angular2html
cd LASA/process_scripts
python augment_arkit_partial_points.py
python augment_synthetic_partial_points.py
```
Finally, run the following command to generate **train/val splits**, please check ./dataset/taxonomy for the sub-cateory definition, and 
run all the categories that you want to use. 
```angular2html
cd LASA/process_scripts
python generate_split_for_arkit.py
python generate_split_for_synthetic_data.py
```

## Raw Dataset processing for reconstruction training
After the **LASA dataset** is downloaded, then download the **ArkitScene 3dod dataset** from <a href="https://github.com/apple/ARKitScenes">ArkitScene's repository</a>. We provide a list of annotated scene in
<a href=https://github.com/GAP-LAB-CUHK-SZ/LASA/arkitscene_process_script/annotate_scene_list.txt>annotate_scene_list.txt</a>. Only these scene need to be downloaded.
You can put the <a href="https://github.com/apple/ARKitScenes/blob/main/download_data.py">download_data.py</a> under **LASA/arkitscene_process_script** and use the following script to download the selected data:
```angular2html
cd LASA/arkitscene_process_script
python donwload_select_arkitscene.py --save_dir <path_to_arkit>
```
Unzip all files by the following commands.
```angular2html
cd LASA/arkitscene_process_script
python unzip_arkit_data.py --arkit_dir <path_to_arkit_dataset/3dod> --split Training
python unzip_arkit_data.py --arkit_dir <path_to_arkit_dataset/3dod> --split Validation
```
The procedure includes several steps. 
- It firstly **select some frames** from ArkitScene datasets.
- Then, **super resolution** is employed to upscale the original low resolution images.
- Then, it will **crop the objects in the image**, and **recompute the camera pose and 
intrinsic matrix** for each image. 
- Finally, convert CAD mesh into watertight mesh, and compute the occupancy for sampling points.

Secondly, select some images such that the object are visible in this images by the following command:
```angular2html
cd LASA/arkitscene_process_script
python select_arkitscene_images.py --arkit_root <path_to_arkit_dataset/3dod> --save_root <path_to_arkit_dataset/images> --split Training
python select_arkitscene_images.py --arkit_root <path_to_arkit_dataset/3dod> --save_root <path_to_arkit_dataset/images> --split Validation
```
Thirdly, install <a href="https://github.com/XPixelGroup/HAT"> HAT <a/> for super resolution on the low resolution images.
Download the pretrained model **HAT_SRx4_ImageNet-pretrain.pth** from <a href="https://github.com/XPixelGroup/HAT"> HAT <a/>, 
and put the checkpoint file under /LASA/checkpoint/SR_model/HAT_SRx4_ImageNet-pretrain.pth. 
Then, run the following command to super resolution the images.
```angular2html
cd LASA/arkitscene_process_script
CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15000 --nproc_per_node=4 \
SR_images.py --image_dir <path_to_arkit_dataset/images>
```
Then, run the following command to crop and pad the images, meanwhile recompute the projection matrix, so
that the points can be projected to the crop images directly using this matrix:
```angular2html
cd LASA/arkitscene_process_script
python crop_arkit_images.py --image_dir <path_to_arkit_dataset/images> --lasa_dir <path_to_LASA_dataset> \
--arkit_dir <path_to_arkit_dataset/3dod> --consider_alignment
```
The mesh annotation is originally aligned with the Laser scan, and might be slightly misaligned with the RGB-D scan.
Therefore, we set **--consider_alignment** flag, which is used to further align the annotation with the RGB-D scan. 
(In most of the cases, RGB-D mesh and laser point clouds are well aligned.)

Next, Install <a href="https://github.com/hjwdzh/Manifold">Manifold</a>, and add the **build** folder to
the **PATH** environment variable. Then, convert the gt mesh into watertight mesh by the following commands:
```angular2html
cd LASA/arkitscene_process_script
python convert_watertight.py --lasa_dir <path_to_LASA_dataset>
```


Next, generate occupancy GT for the annotation, and format the RGB-D and LiDAR point cloud for inputs.
```angular2html
cd LASA/arkitscene_process_script
python compute_occ_and_format_points_data.py --lasa_dir <path_to_LASA_dataset> --consider_alignment
```
In case of running on a headless machine, you can refer to 
<a href="https://pyrender.readthedocs.io/en/latest/install/index.html?highlight=ssh#getting-pyrender-working-with-osmesa">this page</a> for
how to use Mesa for computing the sdf or occupancy values, and remember to set the environment variable **PYOPENGL_PLATFORM="osmesa"**. 
Or try **xvfb** library to create a virtual display. 