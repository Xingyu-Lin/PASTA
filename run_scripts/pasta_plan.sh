# <PATH TO PRETRAINED POINTFLOW MODEL (SET INPUT)>: pointflow model after trained with 'load_set_from_buffer=True'.
# <PATH TO PRETRAINED POLICY MODEL (PASTA AGENT)>: pasta model with policy trained.
# --vae_resume_path '<PATH TO PRETRAINED POINTFLOW MODEL (SET INPUT)>' \
# --resume_path '<PATH TO PRETRAINED POLICY MODEL (PASTA AGENT)>' \

python core/pasta/train_pasta.py \
    --env_name 'CutRearrangeSpread-v1' \
    --input_mode 'pc' \
    --run_plan True \
    --vae_resume_path 'data/released_models/pointflow_vae/CutRearrangeSpread-v1/pointflow_pasta_crs.pt' \
    --resume_path 'data/released_models/pasta_abstraction/CutRearrangeSpread-v1/fullmodel_pasta_crs.ckpt' \
    --train_modules '' \
    --load_modules 'reward' 'fea' 'policy' \
    --plan_step 3 \
    --num_tools 3 \
    --dimz 2 \
    --actor_arch 'pointnet' \
    --actor_latent_dim 64 \
    --reward_latent_dim 1024 \
    --fea_arch 'v3' \
    --fea_latent_dim 1024 \
    --fea_center True \
    --eval_plan True \
    --plan_bs 5000 \
    --adam_sample 5000 \
    --adam_iter 100 \
    --use_wandb 1
