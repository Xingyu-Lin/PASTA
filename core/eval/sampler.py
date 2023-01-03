import time
import numpy as np
import torch
import os
from core.utils import logger

from core.utils.diffskill_utils import get_img, img_to_tensor, to_action_mask, batch_pred
from core.traj_opt.env_spec import get_reset_tool_state
from core.utils.diffskill_utils import img_to_tensor, to_action_mask, batch_pred
from core.pasta.generate_dbscan_label import dbscan_cluster
from core.utils.pasta_utils import match_set_pcl
from core.utils.pc_utils import decompose_pc, resample_pc
from plb.envs.mp_wrapper import SubprocVecEnv

device = 'cuda'


def sample_traj_setagent(env, agent, reset_key, tid, action_mask=None, log_succ_score=False, reset_primitive=False):
    """Compute ious: pairwise iou between each pair of timesteps. """
    assert agent.args.num_env == 1
    states, obses, actions, rewards, succs, scores = [], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores
    if action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        # rgbd observation
        obs = env.render(mode='rgb', img_size=agent.args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    n_in, n_out = agent.skill_def[tid]['in'], agent.skill_def[tid]['out']
    T = 50
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    frame_stack = agent.args.frame_stack
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    if reset_primitive:
        primitive_state = env.getfunc('get_primitive_state', 0)
    if reset_key is not None:
        infos = [mp_info[0]]
    else:
        infos = []

    if agent.args.input_mode == 'rgbd':
        raise NotImplementedError
    else:
        goal_dpc = np.array(env.getattr('target_pc', 0))[None]
        goal_label = dbscan_cluster(goal_dpc, args=agent.args)
        pcs_goal = decompose_pc(goal_dpc, goal_label, N=1000)

    with torch.no_grad():
        # Perform set matching for pointcloud and u, use a higher threshold
        if agent.args.filter_set:
            dpc_start = state[:3000].reshape(-1, 1000, 3)
            label = dbscan_cluster(dpc_start, args=agent.args)
            pcs_obs = decompose_pc(dpc_start, label, N=1000)
            dpc, goal_dpc, info = match_set_pcl(1, [pcs_obs], [pcs_goal], n_in, n_out, dist='chamfer',
                                                chamfer_downsample=200, verbose=False, eval=True, thr=5e-4)
            goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)
            filtered_pcs_goal = [pcs_goal[int(i)] for i in info['pc_idx2'][0][0]]
            filtered_obs_idx = [i for (i, pcl) in enumerate(dpc_start[0]) if pcl in dpc[0]]
            print("filtered goal:", len(filtered_pcs_goal))
            print("len filtered obs:", len(filtered_obs_idx))
        else:
            filtered_obs_idx = np.arange(1000)
            goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)

        for i in range(T):
            t1 = time.time()
            with torch.no_grad():   
                if agent.args.input_mode == 'rgbd':
                    raise NotImplementedError
                else:
                    tool_particles = None
                    dpc = state[:3000].reshape(-1, 1000, 3)
                    dpc = dpc[:, filtered_obs_idx, :]
                    dpc = resample_pc(dpc[0], 1000)[None]

                    state = torch.FloatTensor(state).to(agent.device)[None]
                    dpc = torch.FloatTensor(dpc).to(agent.device)
                    action, done, act_info = agent.act(state, dpc, goal_dpc, tid, tool_particles=tool_particles)
                    if log_succ_score:
                        u_obs, u_goal = act_info['u_obs'], act_info['u_goal']
                        if i == 0:
                            u_init = u_obs.clone()
                        succ = batch_pred(agent.feas[tid], {
                            'obs': u_init, 'goal': u_obs, 'eval': True}).detach().cpu().numpy()[0]
                        score = batch_pred(agent.reward_predictor, {
                            'obs': u_obs, 'goal': u_goal, 'eval': True}).detach().cpu().numpy()[0]
                    action = action[0].detach().cpu().numpy()
                    done = done[0].detach().cpu().numpy()
            if np.round(done).astype(int) == 1 and agent.terminate_early:
                break
            t2 = time.time()
            mp_next_state, mp_reward, _, mp_info = env.step([action])
            next_state, reward, info = mp_next_state[0], mp_reward[0], mp_info[0]

            infos.append(info)
            t3 = time.time()

            agent_time += t2 - t1
            env_time += t3 - t2

            actions.append(action)
            states.append(next_state)
            obs = env.render(
                [{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]
            state = next_state
            obses.append(obs)
            total_r += reward
            rewards.append(reward)
            if log_succ_score:
                succs.append(succ)
                scores.append(score)
        if reset_primitive:
            _, _, reset_obses, _, _ = env.getfunc('primitive_reset_to', 0,
                                                  list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])  # TODO tid
            for obs in reset_obses:
                assert frame_stack == 1
                if log_succ_score:
                    succ = 0
                    score = 0
                obses.append(obs)
                if log_succ_score:
                    succs.append(succ)
                    scores.append(score)

    target_img = np.array(env.getattr('target_img', 0))
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if log_succ_score:
        ret['succs'] = np.array(succs)  # Should miss the first frame
        ret['scores'] = np.array(scores)
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj_agent(env, agent, reset_key, tid, log_succ_score=False, reset_primitive=False):
    """Compute ious: pairwise iou between each pair of timesteps. """
    assert agent.args.num_env == 1
    states, obses, actions, rewards, succs, scores = [], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        # rgbd observation
        obs = env.render(mode='rgb', img_size=agent.args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    T = 50
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
    frame_stack = agent.args.frame_stack
    _, _, _, mp_info = env.step([np.zeros(action_dim)])
    if reset_primitive:
        primitive_state = env.getfunc('get_primitive_state', 0)
    if reset_key is not None:
        infos = [mp_info[0]]
    else:
        infos = []

    if agent.args.input_mode == 'rgbd':
        stack_obs = img_to_tensor(np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)  # stack_obs shape: [1, 4, 64, 64]
        target_img = img_to_tensor(np.array(env.getattr('target_img', 0))[None], mode=agent.args.img_mode).to(agent.device)
        C = stack_obs.shape[1]
        stack_obs = stack_obs.repeat([1, frame_stack, 1, 1])
    else:
        goal_dpc = torch.FloatTensor(np.array(env.getattr('target_pc', 0))[None]).to(agent.device)

    with torch.no_grad():
        for i in range(T):
            t1 = time.time()
            with torch.no_grad():
                if agent.args.input_mode == 'rgbd':
                    obs_tensor = img_to_tensor(
                        np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)
                    stack_obs = torch.cat([stack_obs, obs_tensor], dim=1)[:, -frame_stack * C:]
                    action, done, _ = agent.act_rgbd(stack_obs, target_img, tid)
                    if log_succ_score:
                        z_obs, _, _ = agent.vae.encode(stack_obs)
                        z_goal, _, _ = agent.vae.encode(target_img)
                        if i == 0:
                            z_init = z_obs.clone()
                        succ = batch_pred(agent.feas[tid], {
                            'obs': z_init, 'goal': z_obs, 'eval': True}).detach().cpu().numpy()[0]
                        score = batch_pred(agent.reward_predictor, {
                            'obs': z_obs, 'goal': z_goal, 'eval': True}).detach().cpu().numpy()[0]
                else:
                    state = torch.FloatTensor(state).to(agent.device)[None]
                    action, done, info = agent.act(state, goal_dpc, tid)
                    if log_succ_score:
                        u_obs, u_goal = info['u_obs'], info['u_goal']
                        if i == 0:
                            u_init = u_obs.clone()
                        succ = batch_pred(agent.feas[tid], {
                            'obs': u_init, 'goal': u_obs, 'eval': True}).detach().cpu().numpy()[0]
                        score = batch_pred(agent.reward_predictor, {
                            'obs': u_obs, 'goal': u_goal, 'eval': True}).detach().cpu().numpy()[0]
            action = action[0].detach().cpu().numpy()
            done = done[0].detach().cpu().numpy()
            if np.round(done).astype(int) == 1 and agent.terminate_early:
                break
            t2 = time.time()
            mp_next_state, mp_reward, _, mp_info = env.step([action])
            next_state, reward, info = mp_next_state[0], mp_reward[0], mp_info[0]

            infos.append(info)
            t3 = time.time()

            agent_time += t2 - t1
            env_time += t3 - t2

            actions.append(action)
            states.append(next_state)
            obs = env.render(
                [{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]
            state = next_state
            obses.append(obs)
            total_r += reward
            rewards.append(reward)
            if log_succ_score:
                succs.append(succ)
                scores.append(score)
        if reset_primitive:
            _, _, reset_obses, _, _ = env.getfunc('primitive_reset_to', 0,
                                                  list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])  # TODO tid
            for obs in reset_obses:
                assert frame_stack == 1
                if log_succ_score:
                    if agent.args.input_mode == 'rgbd':
                        with torch.no_grad():
                            z_obs, _, _ = agent.vae.encode(stack_obs)
                            z_goal, _, _ = agent.vae.encode(target_img)
                            if i == 0:
                                z_init = z_obs.clone()
                            succ = batch_pred(agent.feas[tid], {
                                'obs': z_init, 'goal': z_obs, 'eval': True}).detach().cpu().numpy()[0]
                            score = batch_pred(agent.reward_predictor, {
                                'obs': z_obs, 'goal': z_goal, 'eval': True}).detach().cpu().numpy()[0]
                    else:
                        succ = 0
                        score = 0
                obses.append(obs)
                if log_succ_score:
                    succs.append(succ)
                    scores.append(score)

    target_img = np.array(env.getattr('target_img', 0))
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time}
    if log_succ_score:
        ret['succs'] = np.array(succs)  # Should miss the first frame
        ret['scores'] = np.array(scores)
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj_solver(env, agent, reset_key, tid, action_mask=None, reset_primitive=False, primitive_reset_states=None, solver_init='zero',
                       num_moves=1):
    """Compute ious: pairwise iou between each pair of timesteps. """
    assert not isinstance(env, SubprocVecEnv)
    states, obses, actions, rewards = [], [], [], []
    if action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])

    if reset_key is not None:
        state = env.reset(**reset_key)
    if reset_primitive:
        if primitive_reset_states is None:
            primitive_reset_states = []
            primitive_reset_states.append(env.get_primitive_state())
        for idx, prim in enumerate(env.taichi_env.primitives):
            prim.set_state(0, primitive_reset_states[0][idx])

    # rgbd observation
    obs = env.render(mode='rgb', img_size=agent.args.img_size)

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    T = 50
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()

    # Solver
    taichi_env = env.taichi_env
    action_dim = taichi_env.primitives.action_dim
    for _ in range(10):
        state, _, _, info = env.step(np.zeros(action_dim))

    # for trajectory optimization evaluation, need to reset the tool.
    if agent.args.data_name == 'single' and agent.args.env_name == 'CutRearrangeSpread-v1':
        for tid in range(agent.args.num_tools):
            tool_reset_state = get_reset_tool_state('CutRearrangeSpread-v1', state[:3000].reshape(-1, 1000, 3), tid)
            env.taichi_env.primitives[tid].set_state(0, tool_reset_state)


    infos = [info]
    actions = []
    for move in range(num_moves):
        if solver_init == 'multiple':
            init_actions = np.zeros(
                [3, T, taichi_env.primitives.action_dim], dtype=np.float32)  # left, right, zero
            init_actions[0, :10, 3] = -1.  # left
            init_actions[1, :10, 3] = 1.  # right
            all_infos = []
            all_buffers = []
            for i in range(len(init_actions)):
                init_action = init_actions[i]
                cur_info, cur_buffer = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss,
                                                   max_iter=agent.args.gd_max_iter,
                                                   lr=agent.args.lr)
                all_infos.append(cur_info)
                all_buffers.append(cur_buffer)
            improvements = np.array(
                [(all_buffers[i][0]['loss'] - all_infos[i]['best_loss']) / all_buffers[i][0]['loss'] for i in range(len(all_infos))])
            solver_info = all_infos[np.argmax(improvements)]
        else:
            if isinstance(solver_init, float):   # Cut location for CRS
                cut_loc = solver_init
                init_action = np.zeros(
                    [T, taichi_env.primitives.action_dim], dtype=np.float32)
                cur_loc = state[3000]
                act_x = (cut_loc - cur_loc) / 0.015 / 10
                init_action[:10, 0] = act_x
                init_action[10:20, 1] = -1
            elif solver_init == 'zero':
                init_action = np.zeros(
                    [T, taichi_env.primitives.action_dim], dtype=np.float32)
            else:
                raise NotImplementedError
            solver_info, _ = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss, max_iter=agent.args.gd_max_iter,
                                         lr=agent.args.lr)

        agent.save_plot_buffer(os.path.join(logger.get_dir(), f'solver_loss.png'))
        agent.dump_buffer(os.path.join(logger.get_dir(), f'buffer.pkl'))
        solver_actions = solver_info['best_action']
        actions.extend(solver_actions)
        agent_time = time.time() - st_time
        for i in range(T):
            t1 = time.time()
            next_state, reward, _, info = env.step(solver_actions[i])
            infos.append(info)
            env_time += time.time() - t1
            states.append(next_state)
            obs = taichi_env.render(mode='rgb', img_size=agent.args.img_size)
            obses.append(obs)
            total_r += reward
            rewards.append(reward)
        target_img = env.target_img

        if reset_primitive and move < num_moves - 1:
            if len(primitive_reset_states) < num_moves + 1:
                primitive_reset_states.append(primitive_reset_states[-1])
            else:
                assert len(primitive_reset_states) == num_moves + 1
            for idx, prim in enumerate(env.taichi_env.primitives):
                prim.set_state(0, primitive_reset_states[move + 1][idx])

        emds = np.array([info['info_emd'] for info in infos])
        if len(infos) > 0:
            info_normalized_performance = np.array(
                [info['info_normalized_performance'] for info in infos])
            info_final_normalized_performance = info_normalized_performance[-1]
        else:
            info_normalized_performance = []
            info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if reset_key is not None:
        ret.update(**reset_key)
    return ret


def sample_traj_replay(env, reset_key, tid, action_sequence, action_mask=None, img_size=64, save_mode=None, args=None):
    """Compute ious: pairwise iou between each pair of timesteps. """
    def overlay(img1, img2):
        mask = img2[:, :, 3][:, :, None]
        return img1 * (1 - mask) + img2 * mask

    assert not isinstance(env, SubprocVecEnv)
    states, obses, actions, rewards = [], [], [], []
    if action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': img_size}])[
            0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        obs = env.render(mode='rgb', img_size=img_size)  # rgbd observation

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    # Replay trajectories
    action_dim = env.taichi_env.primitives.action_dim
    _, _, _, info = env.step(np.zeros(action_dim))
    infos = [info]
    xs = np.linspace(0., 1., len(action_sequence))
    ys = []
    for i in range(len(action_sequence)):
        t1 = time.time()

        with torch.no_grad():
            action = action_sequence[i]
        t2 = time.time()
        next_state, reward, _, info = env.step(action)
        t3 = time.time()

        agent_time += t2 - t1
        env_time += t3 - t2

        actions.append(action)
        states.append(next_state)
        obs = env.render(mode='rgb', img_size=img_size)
        if save_mode=='plot':
            ys.append(info['info_normalized_performance'])
            img = get_img(args, xs[:len(ys)], ys)
            obs = np.clip(overlay(obs, img), 0., 1.)
        obses.append(obs)
        infos.append(info)
        total_r += reward
        rewards.append(reward)
    target_img = env.target_img
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if reset_key is not None:
        ret.update(**reset_key)
    return ret

def sample_traj_replay_execute(agent, env, plan_traj, reset_key, action_sequence, action_mask=None, img_size=64, save_mode=None):
    """Compute ious: pairwise iou between each pair of timesteps. """
    def overlay(img1, img2):
        mask = img2[:, :, 3][:, :, None]
        return img1 * (1 - mask) + img2 * mask

    assert not isinstance(env, SubprocVecEnv)
    states, obses, actions, rewards = [], [], [], []

    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode': 'rgb', 'img_size': img_size}])[
            0]  # rgbd observation
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        obs = env.render(mode='rgb', img_size=img_size)  # rgbd observation

    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    total_r = 0
    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    # Replay trajectories
    action_dim = env.taichi_env.primitives.action_dim
    _, _, _, info = env.step(np.zeros(action_dim))
    infos = [info]
    xs = np.linspace(0., 1., len(plan_traj['us'])*60)
    ys = []
    for step in range(1, len(plan_traj['traj_img']) - 1):
        t1 = time.time()
        tid = plan_traj['tool'][step - 1]
        print('tool id:', tid)
        if 'resets' in plan_traj.keys() or not agent.args.filter_set:
            filtered_obs_idx = np.arange(1000)
            dpc = np.zeros((1000, 3))
            goal_dpc = torch.FloatTensor(dpc[None]).to(agent.device)
        else:  # have to do a whole bunch of filtering to decide where to reset the tool
            if agent.args.train_set:
                n_in, n_out = agent.skill_def[tid]['in'], agent.skill_def[tid]['out']
                # Getting goal
                pcs_goal = []
                u_goals = plan_traj['us'][step - 1]
                pcs = agent.vae.decode(torch.FloatTensor(u_goals).to(agent.device), 1000).squeeze(0)
                for k in range(pcs.shape[0]):
                    pcs_goal.append(pcs[k].detach().cpu().numpy())
            else:
                n_in = n_out = 1
                u_goal = torch.FloatTensor(plan_traj['zs'][step - 1, :]).to(agent.device)[None].repeat(2, 1)
                tensor_goal = agent.vae.decode(u_goal, num_points=1000)[:1]
                pcs_goal = tensor_goal.detach().cpu().numpy()

            if 'masks_in' in plan_traj.keys():
                mask_in = plan_traj['masks_in'][step - 1]
                dpc_start = state[:3000].reshape(-1, 1000, 3)
                label = dbscan_cluster(dpc_start)
                pcs_obs = decompose_pc(dpc_start, label, N=1000)
                print(len(pcs_obs))
                if step == 1:
                    dpc = pcs_obs[mask_in.item()][None]
                else:
                    intend_u = plan_traj['us'][step - 2][mask_in] # n_in x dimu
                    intended_com = intend_u[:, -3:]
                    idx = np.argmin([np.linalg.norm(np.mean(pc, axis=0) - intended_com) for pc in pcs_obs])  
                    pcs_obs = pcs_obs[idx]
                    dpc = resample_pc(np.vstack(pcs_obs), 1000)[None]
                filtered_obs_idx = [i for (i, pcl) in enumerate(dpc_start[0]) if pcl in dpc[0]]
                goal_dpc = np.vstack(pcs_goal[:n_out])[None]
                goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)
            else:
                filtered_obs_idx = list(range(1000))
                min_samples = agent.args.dbscan_min_samples
                while len(filtered_obs_idx) == 1000:
                    # print("dbscaning, minsamples:", min_samples)
                    # Getting obs
                    dpc_start = state[:3000].reshape(-1, 1000, 3)
                    label = dbscan_cluster(dpc_start, min_samples=min_samples)
                    pcs_obs = decompose_pc(dpc_start, label, N=1000)
                    # Perform set matching for pointcloud and u, use a lower threshold
                    dpc, goal_dpc, info = match_set_pcl(1, [pcs_obs], [pcs_goal], n_in, n_out, dist='chamfer',
                                                        chamfer_downsample=200, verbose=False, eval=True, thr=5e-4)
                    if goal_dpc is None:
                        exec_flag = False
                        break
                    goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)
                    filtered_pcs_goal = [pcs_goal[int(i)] for i in info['pc_idx2'][0][0]]
                    filtered_obs_idx = [i for (i, pcl) in enumerate(dpc_start[0]) if pcl in dpc[0]]
                    # print("filtered goal:", len(filtered_pcs_goal))
                    # print("len filtered obs:", len(filtered_obs_idx))
                    if step == 1:
                        break
                    min_samples += 1
        for i in range(60):
            if i == 0 and 'CutRearrangeSpread' in agent.args.env_name:
                if 'resets' in plan_traj.keys():
                    tool_reset_state = plan_traj['resets'][step-1]
                else:
                    tool_reset_state = get_reset_tool_state(agent.args.env_name, dpc, tid)
                env.taichi_env.primitives[tid].set_state(0, tool_reset_state)
            with torch.no_grad():
                idx = i+(step-1)* (50+10)   # 10 stopping actions
                # if idx == 287:   # done
                #     break
                action = action_sequence[idx]
            t2 = time.time()
            state, reward, _, info = env.step(action)
            t3 = time.time()

            agent_time += t2 - t1
            env_time += t3 - t2

            actions.append(action)
            states.append(state)
            obs = env.render(mode='rgb', img_size=img_size)
            if save_mode=='plot':
                ys.append(info['info_normalized_performance'])
                img = get_img(agent.args, xs[:len(ys)], ys)
                obs = np.clip(overlay(obs, img), 0., 1.)
            obses.append(obs)
            infos.append(info)
            total_r += reward
            rewards.append(reward)

        # for i in range(10):
        #     action = np.zeros(agent.args.action_dim)
        #     state, reward, _, info = env.step(action)
        #     actions.append(action)
        #     states.append(state)
        #     obs = env.render(mode='rgb', img_size=img_size)
        #     obses.append(obs)
        #     infos.append(info)
        #     total_r += reward
        #     rewards.append(reward)
        if 'Cut' in agent.args.env_name:
            tool_reset_state = get_reset_tool_state(agent.args.env_name, dpc, tid, back=True)
            env.taichi_env.primitives[tid].set_state(0, tool_reset_state)

    target_img = env.target_img
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array(
            [info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if reset_key is not None:
        ret.update(**reset_key)
    return ret




def sample_traj(env, agent, reset_key, tid, action_mask=None, action_sequence=None, log_succ_score=False, reset_primitive=False, solver_init='zero',
                num_moves=1):
    print("This function is no longer being used. Please call one of `sample_traj_agent`, `sample_traj_solver`, or `sample_traj_reply` instead.")
    raise NotImplementedError
