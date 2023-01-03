from tqdm import tqdm
import numpy as np
from plb.envs.mp_wrapper import SubprocVecEnv


def generate_reset_motion(buffer, env, horizon=50, max_reset_length=20, **kwargs):
    """ For each trajectory in the buffer, generate the trajectory for reset motion.
        If reset motion length > max_reset_length, then just subsample them
    """
    num_trajs = len(buffer) // horizon

    obs = buffer.buffer['obses'][0]
    state_shape, state_dtype = buffer.buffer['states'][0].shape, buffer.buffer['states'][0].dtype 
    obs_shape, obs_dtype = obs.shape, obs.dtype

    reset_motion_states = np.zeros([num_trajs, max_reset_length, *state_shape], dtype=state_dtype)
    reset_motion_obses = np.zeros([num_trajs, max_reset_length, *obs_shape], dtype=obs_dtype)
    reset_info_emds = np.zeros([num_trajs, max_reset_length], dtype=np.float32)
    reset_motion_lens = np.zeros(shape=(num_trajs,), dtype=np.int32)

    for traj_id in tqdm(range(num_trajs), desc='gen_reset_motion'):
        traj_t = traj_id * horizon
        init_v, target_v = buffer.buffer['init_v'][traj_t], buffer.buffer['target_v'][traj_t]
        if isinstance(env, SubprocVecEnv):
            env.reset([{'init_v': init_v, 'target_v': target_v}])
            demo_actions = buffer.buffer['actions'][traj_t:traj_t + horizon]
            primitive_states = env.getfunc('get_primitive_state', 0)
            for action in demo_actions:
                env.step([action])
            tid = buffer.get_tid(buffer.buffer['action_mask'][traj_t])
            states, _, obses, _, infos = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': tid, 'reset_states': primitive_states}])
        else:
            env.reset(init_v=init_v, target_v=target_v)
            demo_actions = buffer.buffer['actions'][traj_t:traj_t + horizon]
            primitive_states = env.get_primitive_state()
            for action in demo_actions:
                env.step(action)
            tid = buffer.get_tid(buffer.buffer['action_mask'][traj_t])
            states, _, obses, _, infos = env.primitive_reset_to(idx=tid, reset_states=primitive_states, **kwargs)

        if len(obses) == 0:
            continue
        emds = np.array([info['info_emd'] for info in infos])
        if len(obses) > max_reset_length:
            idx = sorted(np.random.choice(range(len(obses)), max_reset_length, replace=False))
            states = states[idx]
            obses = obses[idx]
            emds = emds[idx]

        reset_motion_states[traj_id, :len(states)] = states
        reset_motion_obses[traj_id, :len(obses)] = obses
        reset_info_emds[traj_id, :len(obses)] = emds
        reset_motion_lens[traj_id] = len(obses)
        print('info emds:', emds)

    buffer.buffer['reset_motion_states'] = reset_motion_states
    buffer.buffer['reset_motion_obses'] = reset_motion_obses
    buffer.buffer['reset_motion_lens'] = reset_motion_lens
    buffer.buffer['reset_info_emds'] = reset_info_emds


# Used for launch experiments to make after-gen remedy


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from core.diffskill.args import get_args
    from plb.envs.mp_wrapper import make_mp_envs
    args = get_args(cmd=False)
    args.env_name = 'CutRearrange-v1'
    env = make_mp_envs(args.env_name, args.num_env, args.seed)
    dataset_path = arg_vv['dataset_path']
    from core.diffskill.imitation_buffer import ImitationReplayBuffer
    import os
    buffer = ImitationReplayBuffer(args=None)
    buffer.load(dataset_path)
    generate_reset_motion(buffer, env)
    buffer.save(os.path.join(log_dir, 'dataset.gz'))
    # buffer2 = ImitationReplayBuffer(args=None)
    # buffer2.load(os.path.join(log_dir, 'dataset.gz'))
    # breakpoint()

if __name__ == '__main__':



    arg_vv = {
        'dataset_path': 'datasets/0202_cutrearrange_dataset/0202_cutrearrange_dataset_2022_02_28_21_37_35_0001'
    }
    run_task(arg_vv, 'data/debug', exp_name='')