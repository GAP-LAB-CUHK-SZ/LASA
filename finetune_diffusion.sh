cd scripts
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --master_port 15003 --nproc_per_node=8 \
train_triplane_diffusion.py \
--configs ../configs/finetune_triplane_diffusion.yaml \
--accum_iter 2 \
--output_dir ../output/finetune_dm/lowres_chair \
--log_dir ../output/finetune_dm/lowres_chair --num_workers 8 \
--batch_size 22 \
--blr 1e-4 \
--epochs 500 \
--dist_eval \
--warmup_epochs 20 \
--ae-pth ../output/ae/chair/best-checkpoint.pth \
--category chair \
--finetune \
--finetune-pth ../output/dm/chair/best-checkpoint.pth \
--data-pth ../data \
--replica 5