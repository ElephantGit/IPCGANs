CUDA_VISIBLE_DEVICES=0 python train.py \
  --gan_loss_weight=75 
  --fea_loss_weight=0.5e-4 \
  --age_loss_weight=60 \
  --fea_layer_name=conv5 \
  --checkpoint_dir=./checkpoints/0_conv5_lsgan_transfer_g75_0.5f-4_a60 \
  --sample_dir=age/0_conv5_lsgan_transfer_g75_0.5f-4_a60 \
  --source_file=data/megaage_asian/source_file.txt \
  --root_folder=data/megaage_asian/test/
