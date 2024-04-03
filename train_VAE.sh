cd scripts
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --master_port 15000 --nproc_per_node=8 \
train_triplane_vae.py \
--configs ../configs/train_triplane_vae.yaml \
--accum_iter 2 \
--output_dir ../output/ae/chair \
--log_dir ../output/ae/chair --num_workers 8 \
--batch_size 22 \
--epochs 200 \
--warmup_epochs 5 \
--dist_eval \
--clip_grad 0.35 \
--category chair \
--data-pth ../data \
--replica 5