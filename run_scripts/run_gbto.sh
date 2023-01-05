python core/traj_opt/gen_data.py \
    --env_name 'CutRearrange-v1' \
    --algo 'imitation' \
    --adam_loss_type 'twoway_chamfer' \
    --data_name 'demo' \
    --gen_num_batch 1 \
    --gen_batch_id 0 \
    --gd_max_iter 5