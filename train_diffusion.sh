cd scripts
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --master_port 15004 --nproc_per_node=8 \
train_triplane_diffusion.py \
--configs ../configs/train_triplane_diffusion.yaml \
--accum_iter 2 \
--output_dir ../output/dm/debug \
--log_dir ../output/dm/debug \
--num_workers 8 \
--batch_size 22 \
--epochs 1000 \
--dist_eval \
--warmup_epochs 40 \
--category chair \
--ae-pth ../output/ae/chair/best-checkpoint.pth \
--data-pth ../data \
--replica 5