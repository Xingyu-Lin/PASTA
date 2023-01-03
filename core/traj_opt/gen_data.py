from core.eval.sampler import sample_traj_solver
from core.diffskill.imitation_buffer import ImitationReplayBuffer
from core.diffskill.args import get_args
from core.utils.diffskill_utils import visualize_dataset
from core.utils.diffskill_utils import visualize_trajs
from core.traj_opt.env_spec import set_render_mode, get_tool_spec
from core.traj_opt.env_spec import check_correct_tid
from core.utils.core_utils import set_random_seed
from plb.envs import make
import pickle
import random
import numpy as np
import torch
import json
import os
from core.utils import logger

def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from plb.engine.taichi_env import TaichiEnv
    from core.traj_opt.solver import Solver
    args = get_args(cmd=False)
    args.__dict__.update(**arg_vv)
    print(args.__dict__)

    set_random_seed(args.seed)

    device = 'cuda'

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ----------preparation done------------------
    buffer = ImitationReplayBuffer(args)
    obs_channel = len(args.img_mode) * args.frame_stack

    env = make(args.env_name, nn=(args.algo == 'nn'))
    env.seed(args.seed)
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    set_render_mode(env, args.env_name, 'mesh')

    if args.data_name == 'demo':
        if isinstance(args.num_trajs, tuple):
            traj_ids = np.array_split(np.arange(*args.num_trajs), args.gen_num_batch)[args.gen_batch_id]
        else:
            traj_ids = np.array_split(np.arange(args.num_trajs), args.gen_num_batch)[args.gen_batch_id]
        tids = list(range(args.num_tools))
        def get_state_goal_id(traj_id):
            if 'CutRearrange' in args.env_name:
                np.random.seed(traj_id)
                state_id = traj_id
                goal_id = state_id
            else:
                # Random selection for other env
                np.random.seed(traj_id)
                goal_id = np.random.randint(0, env.num_targets)
                state_id = np.random.randint(0, env.num_inits)
            return {'init_v': state_id, 'target_v': goal_id}  # state and target version
    else:
        tids = [args.tool_combo_id]
        from core.eval.hardcoded_eval_trajs import get_eval_traj
        init_vs, target_vs = get_eval_traj(env.cfg.cached_state_path, plan_step = args.plan_step)
        traj_ids = range(len(init_vs))

        def get_state_goal_id(traj_id):
            return {'init_v': init_vs[traj_id], 'target_v': target_vs[traj_id]}  # state and target version

    solver = Solver(args, taichi_env, (0,), return_dist=True)
    args.dataset_path = os.path.join(logger.get_dir(), 'dataset.gz')

    tool_spec = get_tool_spec(env, args.env_name)    
    for tid in tids:  # Only use the first two tools
        action_mask = tool_spec['action_masks'][tid]
        contact_loss_mask = tool_spec['contact_loss_masks'][tid]
        for i, traj_id in enumerate(traj_ids):
            reset_key = get_state_goal_id(traj_id)
            if args.data_name == 'demo' and not check_correct_tid(args.env_name, reset_key['init_v'], reset_key['target_v'], tid):
                continue

            reset_key['contact_loss_mask'] = contact_loss_mask
            solver_init = 'zero'
            if reset_key['init_v'] % 3 == 1 and args.env_name == 'CutRearrange-v1':
                solver_init = 'multiple'
            if reset_key['init_v'] < 200 and args.env_name == 'CutRearrangeSpread-v1':
                with open(os.path.join(env.cfg.cached_state_path, 'init/cut_loc.pkl'), 'rb') as f:
                    all_cutloc = pickle.load(f)
                cut_loc = all_cutloc[str(reset_key['init_v'])]
                solver_init = cut_loc

            traj = sample_traj_solver(env, solver, reset_key, tid, action_mask=action_mask, solver_init=solver_init)
            print(
                f"traj {traj_id}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
            buffer.add(traj)

            if i % 10 == 0:
                buffer.save(os.path.join(args.dataset_path))
                visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                                  visualize_reset=False,
                                  overlay_target=True)
    if args.data_name == 'demo' and args.env_name != 'CutRearrangeSpread-v1':
        from core.traj_opt.generate_reset_motion import generate_reset_motion
        generate_reset_motion(buffer, env)
        buffer.save(os.path.join(args.dataset_path))
        visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'), visualize_reset=True,
                          overlay_target=True)
    else:
        buffer.save(os.path.join(args.dataset_path))
        visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'), visualize_reset=False,
                          overlay_target=True)

if __name__ == '__main__':
    args = get_args(cmd=True)
    from core.traj_opt.env_spec import get_num_traj
    vel_loss_weights = {
    'LiftSpread-v1': 0.02,
    'GatherMove-v1': 0.0,
    'CutRearrange-v1': 0.02
    }
    lrs = {
        'LiftSpread-v1': 0.02,
        'GatherMove-v1': 0.02,
        'CutRearrange-v1': 0.01
    }
    args.lr = lrs[args.env_name]
    args.vel_loss_weight = vel_loss_weights[args.env_name]
    args.num_trajs = get_num_traj(args.env_name)
    arg_vv = vars(args)
    log_dir = 'data/gbto'
    exp_name = '0001_test_gbto'
    run_task(arg_vv, log_dir, exp_name)