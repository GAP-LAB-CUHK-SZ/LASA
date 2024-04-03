CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15002 --nproc_per_node=2 \
evaluate_object_reconstruction.py \
--configs ../configs/finetune_triplane_diffusion.yaml \
--category arkit_chair arkit_stool \
--ae-pth ../output/ae/chair/best-checkpoint.pth \
--dm-pth ../output/finetune_dm/lowres_chair/best-checkpoint.pth \
--output_folder ../output_result/chair_result \
--data-pth ../data \
--eval_cd \
--reso 256 \
--save_mesh \
--save_par_points \
--save_image \
--save_surface

#check ./datasets/taxonomy to see how sub categories are defined
