CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15002 --nproc_per_node=2 \
export_triplane_features.py \
--configs ../configs/train_triplane_vae.yaml \
--batch_size 10 \
--ae-pth ../output/ae/chair/best-checkpoint.pth \
--data-pth ../data \
--category arkit_chair 03001627 future_chair arkit_stool future_stool ABO_chair
 #sub category