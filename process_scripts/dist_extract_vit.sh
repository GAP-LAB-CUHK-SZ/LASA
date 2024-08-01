CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun --master_port 15000 --nproc_per_node=2 \
extract_img_vit_features.py \
--batch_size 24 \
--ckpt_path ../data/open_clip_pytorch_model.bin