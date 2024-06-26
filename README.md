# LASA
Repository of LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset accepted by CVPR 2024

[<a href="https://arxiv.org/abs/2312.12418">paper</a>]/[<a href="https://gap-lab-cuhk-sz.github.io/LASA/">Project page</a>]

## Demo Results
![292080623-3372c2d9-c788-49de-af62-4d90d2d8468e](https://github.com/GAP-LAB-CUHK-SZ/LASA/assets/40767265/51397fbb-e7bc-44ce-ada9-e9d7f81842ae)
![292080628-a4b020dc-2673-4b1b-bfa6-ec9422625624](https://github.com/GAP-LAB-CUHK-SZ/LASA/assets/40767265/7a0dfc11-5454-428f-bfba-e8cd0d0af96e)
![292080638-324bbef9-c93b-4d96-b814-120204374383](https://github.com/GAP-LAB-CUHK-SZ/LASA/assets/40767265/ee07691a-8767-4701-9a32-19a70e0e240a)

## Dataset
Please fill in the <a href="https://docs.google.com/forms/d/e/1FAIpQLSfKhLLcQ9SA_0yalBzt3SllRg2f4P8uFcAGY7ytDHAsDPg_NA/viewform?usp=sf_link">application form</a> to access raw data of LASA dataset. (link has been updated since 2rd, July)
<br> The dataset is organized as follows: <br>
```
sceneid/
├── sceneid_faro_aligned_clean_0.04.ply # Cleaned and aligned laser scan of the scene
├── sceneid_arkit_mesh.ply             	# TSDF-based mesh reconstruction of the scene
├── sceneid_bbox.npy                    # Bounding box information for the scene
└── instances/
    └── cadid/
        ├── catid_rgbd_mesh.ply         # TSDF-based mesh reconstruction of the instance
        ├── catid_gt_mesh_2.obj         # Watertight Ground truth mesh of the instance
        ├── catid_laser_pcd.ply    			# Point cloud of the instance from laser
        └── catid_laser_pcd_mask.ply 		# Mask for the laser point cloud of the instance
```
The script of generating the preprocessed data for training will be released later

## Download preprocessed data and processing
Download the preprocessed data from <a href="https://pan.baidu.com/s/1X6k82UNG-1hV_FIthnlwcQ?pwd=r7vs">
BaiduYun (code: r7vs)<a/>. (I will update other download methods) Put all the downloaded data under LASA, unzip the align_mat_all.zip mannually. 
You can choose to use the script ./process_scripts/unzip_all_data to unzip all the data in occ_data and other_data by following commands:
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
```

## Evaluation
Download the pretrained weight for each category from <a href="https://pan.baidu.com/s/10liUOaC4CXGn7bN6SQkZsw?pwd=hlf9"> checkpoint.<a/> (code:hlf9). 
Put these folder under LASA/output.<br> The ae folder stores the VAE weight, dm folder stores the diffusion model trained on synthetic data.
finetune_dm folder stores the diffusion model finetuned on LASA dataset.
Run the following commands to evaluate and extract the mesh:
```angular2html
cd evaluation
bash dist_eval.sh
```
The category entries are the sub-category from arkit scenes, please see ./datasets/taxonomy.py about how they are defined.
For example, if you want to evaluate on LASA's chair, category should contain both arkit_chair and arkit_stool. 
make sure the --ae-pth and --dm-pth entry points to the correct checkpoint path. If you are evaluating on LASA,
make sure the --dm-pth points to the finetuned weight in the ./output/finetune_dm folder. The result will be saved
under ./output_result.

## Training
Run the <strong>train_VAE.sh</strong> to train the VAE model. The --category entry in the script specify which category to train on. If you aims to train on one category, just specify one category from <strong> chair, 
cabinet, table, sofa, bed, shelf</strong>. Inputting <strong>all</strong> will train on all categories. Makes sure to download and preprocess all 
the required sub-category data. The sub-category arrangement can be found in ./datasets/taxonomy.py <br>
After finish training the VAE model, run the following commands to pre-extract the VAE features for every object:
```angular2html
cd process_scripts
bash dist_export_triplane_features.sh
```
Then, we can start training the diffusion model on the synthetic dataset by running the <strong>train_diffusion.sh</strong>.<br>
Finally, finetune the diffusion model on LASA dataset by running <strong> finetune_diffusion.sh</strong>. <br><br>

Early stopping is used by mannualy stopping the training by 150 epochs and 500 epochs for training VAE model and diffusion model respetively.
All experiments in the paper are conducted on 8 A100 GPUs with batch size = 22.

## Demo
We prepare a RGBD scan data obtained using iPhone Arkit, which also output object detection results. 
Firstly download example_1.zip from <a href="https://pan.baidu.com/s/1X6k82UNG-1hV_FIthnlwcQ?pwd=r7vs">
BaiduYun (code: r7vs)<a/>. Then unzip it and put the example_1 folder at ./example_data/example_1 <br>
Then, run the following commands to run the demo:
```angular2html
cd demo
bash run_demo.sh
```
The results will be saved in ../example_output_data/example_1 further. <br>
We will further develop a more user-friendly demo.

## TODO

- [ ] Object Detection Code
- [x] Code for Demo on both arkitscene and in the wild data

## Citation
```
@article{liu2023lasa,
  title={LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset},
  author={Liu, Haolin and Ye, Chongjie and Nie, Yinyu and He, Yingfan and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2312.12418},
  year={2023}
}
```
