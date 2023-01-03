# <PATH TO DEMONSTRATION DATASET>: dataset path after generating dbscan labels for the dataset for pasta training
# <PATH TO PRETRAINED POINTFLOW MODEL (SET INPUT)>: pointflow model after trained with 'load_set_from_buffer=True'.
# <PATH TO PRETRAINED POLICY MODEL (PASTA AGENT)>: pasta model with policy trained.
# --dataset_path '<PATH TO DEMONSTRATION DATASET>' \
# --vae_resume_path '<PATH TO PRETRAINED POINTFLOW MODEL (SET INPUT)>' \
# --resume_path '<PATH TO PRETRAINED POLICY MODEL (PASTA AGENT)>' \

python core/pasta/train_pasta.py \
    --env_name 'CutRearrangeSpread-v1' \
    --input_mode 'pc' \
    --dataset_path 'data/released_datasets/demonstrations/CutRearrangeSpread-v1/0609_crs_dataset' \
    --vae_resume_path 'data/released_models/pointflow_vae/CutRearrangeSpread-v1/pointflow_pasta_crs.pt' \
    --resume_path 'data/released_models/pasta_policy/policy_pasta_cutrearrange.ckpt' \
    --train_modules 'reward' 'fea' \
    --load_modules 'policy' \
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
    --fea_z_noise 0.02 \
    --fea_t_noise 0.01 \
    --il_lr 1e-4 \
    --il_eval_freq 10 \
    --il_num_epoch 300 \
    --actor_arch 'pointnet' \
    --fea_arch 'v3' \
    --hard_negative_type 'obs_goal' \
    --eval_plan True \
    --plan_bs 5000 \
    --adam_sample 5000 \
    --adam_iter 100 \
    --use_wandb 1
