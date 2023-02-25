# <PATH TO DEMONSTRATION DATASET>: dataset path after generating dbscan labels for the dataset for pasta training
# <PATH TO PRETRAINED POINTFLOW MODEL (SET INPUT)>: pointflow model after trained with 'load_set_from_buffer=True'

python core/pasta/train_pasta.py \
    --env_name 'CutRearrangeSpread-v1' \
    --input_mode 'pc' \
    --dataset_path 'data/released_datasets/demonstrations/CutRearrangeSpread-v1/0609_crs_dataset' \
    --vae_resume_path 'data/released_models/pointflow_vae/CutRearrangeSpread-v1/pointflow_pasta_crs.pt' \
    --train_modules 'policy' \
    --plan_step 3 \
    --num_tools 3 \
    --dimz 2 \
    --batch_size 256 \
    --actor_batch_size 10 \
    --actor_latent_dim 64 \
    --reward_latent_dim 1024 \
    --fea_latent_dim 1024 \
    --num_random_neg 2048 \
    --num_buffer_neg 2048 \
    --il_eval_freq 100 \
    --il_num_epoch 2000 \
    --eval_skill True \
    --obs_noise 0.005 \
    --il_lr 1e-4 \
    --actor_arch 'pointnet' 
