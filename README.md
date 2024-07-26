# LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset
<div align="center">
  <a href="https://gap-lab-cuhk-sz.github.io/LASA/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2312.12418"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
</div>

![292080623-3372c2d9-c788-49de-af62-4d90d2d8468e](https://github.com/GAP-LAB-CUHK-SZ/LASA/assets/40767265/51397fbb-e7bc-44ce-ada9-e9d7f81842ae)


## Dataset
Please fill in the <a href="https://docs.google.com/forms/d/e/1FAIpQLSfKhLLcQ9SA_0yalBzt3SllRg2f4P8uFcAGY7ytDHAsDPg_NA/viewform?usp=sf_link">application form</a> 
to access raw data of LASA dataset. (link and data has been updated since 24th, July)
<br> The dataset is organized as follows: <br>
```
sceneid/
├── sceneid_faro_aligned_clean_0.04.ply # Cleaned and aligned laser scan of the scene
├── sceneid_arkit_mesh.ply             	# TSDF-based mesh reconstruction of the scene
├── sceneid_arkit_neus.ply(coming)      # NeuS-based mesh reconstruction of the scene
├── sceneid_arkit_gs.ply(coming)        # Gaussian Splatting reconstruction of the scene 
├── sceneid_bbox.npy                    # Bounding box information of the scene
├── sceneid_layout.json(coming)         # Layout Annotation of the scene
└── instances/
    └── cadid/
        ├── cadid_rgbd_mesh.ply         # TSDF-based mesh reconstruction of the instance
        ├── cadid_gt_mesh.obj           # Artist-made Ground Truth mesh of the instance, aligned with laser 
        ├── cadid_gt_mesh_2.obj         # Watertight Ground Truth mesh of the instance, aligned with laser
        ├── cadid_laser_pcd.ply    	    # Point cloud of the instance from laser
        └── alignment.txt 		          # An alignment matrix that align annotation to rgbd mesh
```
Data preprocessing and preparation can be found in <a href="https://github.com/GAP-LAB-CUHK-SZ/LASA/blob/main/arkitscene_process_script/DATA.md">DATA.md</a>.
We also provide preprocessed data for download.


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
