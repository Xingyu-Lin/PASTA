#! /bin/bash
# Number of GPUs used for parallelized training in the paper: 10

dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=2
batch_size=1024
tr_max_sample_points=1000
te_max_sample_points=1000
lr=2e-3
epochs=11
ds=dynabs
env_name='CutRearrangeSpread-v1'
data_dirs='<PATH TO DEMONSTRATION DATASET>'
log_name="0001_set_cutrearrangespread_pointflow"

python PointFlow/train.py \
    --log_name ${log_name} \
    --env_name ${env_name} \
    --lr ${lr} \
    --train_T False \
    --dataset_type ${ds} \
    --cached_state_path '<PATH TO INIT TARGET CONFIGURATIONS>' \
    --data_dirs ${data_dirs} \
    --tr_max_sample_points=${tr_max_sample_points}\
    --te_max_sample_points=${te_max_sample_points}\
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 1 \
    --viz_freq 1 \
    --log_freq 1 \
    --val_freq 1 \
    --distributed \
    --use_latent_flow \
    --load_set_from_buffer True

echo "Done"
exit 0
