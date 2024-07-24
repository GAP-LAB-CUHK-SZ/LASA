for category in {"chair","sofa","table","cabinet","shelf","bed"}
do
CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15002 --nproc_per_node=2 \
evaluate_object_reconstruction.py \
--configs ../configs/finetune_triplane_diffusion.yaml \
--category $category \
--ae-pth ../output/ae/$category/best-checkpoint.pth \
--dm-pth ../output/finetune_dm/$category/best-checkpoint.pth \
--output_folder ../output_result/$category_result \
--data-pth ../data \
--eval_cd \
--reso 256 \
--save_mesh \
--save_par_points \
--save_image \
--save_surface
done
#check ./datasets/taxonomy to see how sub categories are defined
