# LASA
Repository of LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset accepted by CVPR 2024

[<a href="https://arxiv.org/abs/2312.12418">paper</a>]/[<a href="https://gap-lab-cuhk-sz.github.io/LASA/">Project page</a>]

## Demo Results
![292080623-3372c2d9-c788-49de-af62-4d90d2d8468e](https://github.com/GAP-LAB-CUHK-SZ/LASA/assets/40767265/51397fbb-e7bc-44ce-ada9-e9d7f81842ae)

## Dataset
Please fill in the <a href="https://docs.google.com/forms/d/e/1FAIpQLSfKhLLcQ9SA_0yalBzt3SllRg2f4P8uFcAGY7ytDHAsDPg_NA/viewform?usp=sf_link">application form</a> 
to access raw data of LASA dataset. (link and data has been updated since 24th, July)
<br> The dataset is organized as follows: <br>
```
sceneid/
├── sceneid_faro_aligned_clean_0.04.ply # Cleaned and aligned laser scan of the scene
├── sceneid_arkit_mesh.ply             	# TSDF-based mesh reconstruction of the scene
├── sceneid_bbox.npy                    # Bounding box information for the scene
└── instances/
    └── cadid/
        ├── cadid_rgbd_mesh.ply         # TSDF-based mesh reconstruction of the instance
        ├── cadid_gt_mesh_2.obj         # Ground truth mesh of the instance, aligned with laser
        ├── cadid_laser_pcd.ply    	    # Point cloud of the instance from laser
        └── alignment.txt 		        # An alignment matrix that align annotation to rgbd mesh
```
Data preprocessing and preparation can be found in <a href="https://github.com/GAP-LAB-CUHK-SZ/LASA/blob/main/arkitscene_process_script/DATA.md">DATA.md</a>.
We also provide preprocessed data for download.

## Installation
The following steps have been tested on Ubuntu20.04.
- You must have an NVIDIA graphics card with at least 12GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- Install `PyTorch==2.3.0` and `torchvision==0.18.0`.
```sh
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

- Install other dependencies:

```sh
pip install -r requirements.txt
```

## Evaluation
Download the pretrained weight for each category from <a href="https://pan.baidu.com/s/10liUOaC4CXGn7bN6SQkZsw?pwd=hlf9"> checkpoint BaiduYun<a/> (code:hlf9) or 
<a href="https://cuhko365.sharepoint.com/:f:/s/CUHKSZ_SSE_GAP-Lab2/EiqBn0E9VANPmo0h0DMuSOUBJpR_Cy6rHIvDzlz169pcBA?e=Kd8TTz"> checkpoint SharePoint<a/>. 
Put these folder under LASA/output.<br> The ae folder stores the VAE weight, dm folder stores the diffusion model trained on synthetic data.
finetune_dm folder stores the diffusion model finetuned on LASA dataset. Only the ae and finetune_dm is needed for final evaluation.
Run the following commands to evaluate and extract the mesh:
```angular2html
cd evaluation
bash dist_eval.sh
```
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
@inproceedings{liu2024lasa,
  title={LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset},
  author={Liu, Haolin and Ye, Chongjie and Nie, Yinyu and He, Yingfan and Han, Xiaoguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20454--20464},
  year={2024}
}
```
