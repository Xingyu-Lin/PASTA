from email.policy import default
import random
import numpy as np
import torch
import argparse


def get_args(cmd=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=bool, default=False)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument('--env_name', type=str, default='LiftSpread-v1')
    parser.add_argument('--num_env', type=int, default=1)  # Number of parallel environment
    parser.add_argument('--dataset_path', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)

    # Env
    parser.add_argument("--dimtool", type=int, default=8)  # Dimension for representing state of the tool. 8 to incorporate the parallel gripper
    parser.add_argument("--img_size", type=int, default=64)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.02)  # For the solver
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--num_trajs", type=int, default=1000)  # Number of demonstration trajectories
    parser.add_argument("--energy_weight", type=float, default=0.)
    parser.add_argument("--vel_loss_weight", type=float, default=0.)
    parser.add_argument("--solver_init", type=str, default='zero', choices=['zero', 'normal', 'uniform'])
    parser.add_argument("--adam_loss_type", type=str, default="emd")
    parser.add_argument("--stop_action_n", type=int, default=0)
    parser.add_argument("--algo", type=str, default='nn')
    parser.add_argument("--data_name", type=str, default='demo', choices=['demo', 'single'])
    parser.add_argument("--gd_max_iter", type=int, default=200, help="steps for the gradient descent(gd) expert")
    parser.add_argument("--gen_num_batch", type=int, default=1, help="number of machines to perform gbto")
    parser.add_argument("--gen_batch_id", type=int, default=0, help="machine id")

    # Architecture
    parser.add_argument("--input_mode", type=str, default='pc', choices=['rgbd', 'pc'])
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--dimz", type=int, default=8)  # Maybe try multiple values
    parser.add_argument("--actor_feature_dim", type=int, default=64)
    parser.add_argument("--bin_succ", type=bool, default=False)
    parser.add_argument("--vae_beta", type=float, default=1.)
    # parser.add_argument("--vae_actor_noise", type=bool, default=True, help="Add noises to samples of VAE for the actor")
    parser.add_argument("--actor_latent_dim", type=int, default=1024, help="Number of latent nodes for the actor")
    parser.add_argument("--reward_latent_dim", type=int, default=1024, help="Number of latent nodes for the reward predictor")
    parser.add_argument("--fea_latent_dim", type=int, default=1024, help="Number of latent nodes for the feasibility predictor")

    # Actor
    parser.add_argument("--actor_arch", type=str, default='v0')
    parser.add_argument("--actor_z_noise", type=float, default=0.0)
    parser.add_argument("--actor_t_noise", type=float, default=0.0)

    # Training
    parser.add_argument("--il_num_epoch", type=int, default=500)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=5)
    parser.add_argument("--rgb_vae_lr", type=float, default=1e-3)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--step_per_epoch", type=int, default=5000000)
    parser.add_argument("--step_warmup", type=int, default=2000)
    parser.add_argument("--hindsight_goal_ratio", type=float, default=0.5)
    parser.add_argument("--debug_overfit_test", type=bool, default=False)
    parser.add_argument("--obs_noise", type=float, default=0.005)  # Noise for the point cloud in the original space
    parser.add_argument("--vae_resume_path", default=None, help="For loading pretrained vae model")
    parser.add_argument("--num_tools", type=int, default=2)
    parser.add_argument("--implicit_fea", type=bool, default=False)
    parser.add_argument("--weight_reward_predictor", type=float, default=1.)
    parser.add_argument("--weight_vae", type=float, default=0.01)
    parser.add_argument("--weight_fea", type=float, default=10.)
    parser.add_argument("--weight_actor", type=float, default=1.)
    parser.add_argument("--back_prop_encoder", type=bool, default=False)
    parser.add_argument("--eval_plan", type=bool, default=1)
    parser.add_argument("--eval_skill", type=bool, default=1)
    parser.add_argument("--eval_vae", type=bool, default=0)
    parser.add_argument("--eval_single", type=bool, default=0)
    parser.add_argument("--train_modules", nargs='*', default=['reward', 'policy', 'fea'])  # Modules to train
    parser.add_argument("--load_modules", nargs='*', default=['reward', 'policy', 'fea'])  # Modules to load
    parser.add_argument('--filter_buffer', type=bool, default=False)
    parser.add_argument('--rgb_vae_noise', type=float, default=0.01)

    # Feasibility
    parser.add_argument("--fea_type", type=str, default='regression', help="[regression, ifea, sgld]")
    parser.add_argument("--t_relative", type=bool, default=True)

    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--pos_reset_ratio", type=float, default=0.2)  # 20% of the positive goals will come from the reset motion
    parser.add_argument("--num_buffer_neg", type=int, default=64)
    parser.add_argument("--num_random_neg", type=int, default=64)
    parser.add_argument("--fea_z_noise", type=float, default=0.2)
    parser.add_argument("--fea_t_noise", type=float, default=0.01)

    # Plan
    parser.add_argument("--run_plan", type=bool, default=False)
    parser.add_argument("--freeze_z", type=bool, default=False)
    parser.add_argument("--opt_mode", type=str, default='adam')
    parser.add_argument("--adam_sample", type=int, default=5000)
    parser.add_argument("--adam_iter", type=int, default=100)
    parser.add_argument("--adam_lr", type=float, default=1e-2)
    parser.add_argument("--min_zlogl", type=float, default=-30)
    parser.add_argument("--save_goal_his", type=bool, default=False)
    parser.add_argument("--plan_step", type=int, default=2)
    parser.add_argument("--oracle_mode", type=bool, default=False)
    parser.add_argument("--visualize_adam_pc", type=bool, default=False)
    parser.add_argument("--plan_bs", type=int, default=1024, help="Batch size during planning")
    parser.add_argument("--run_plan", type=bool, default=False) # If run_plan, will not do training

    # DBScan
    parser.add_argument("--dbscan_eps", type=float, default=0.03)
    parser.add_argument("--dbscan_min_samples", type=int, default=6)
    parser.add_argument("--dbscan_min_points", type=int, default=10)
    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args
