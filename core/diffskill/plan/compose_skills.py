import pickle
import copy
import numpy as np
import torch
from core.traj_opt.env_spec import get_reset_tool_state
from core.utils.diffskill_utils import write_number
from core.pasta.generate_dbscan_label import dbscan_cluster
from core.utils.pasta_utils import match_set_pcl
from core.utils.pc_utils import decompose_pc
from core.utils.plb_utils import save_numpy_as_gif
from core.utils.diffskill_utils import traj_gaussian_logl
from core.utils.open3d_utils import visualize_point_cloud_plt
import time

from core.diffskill.plan.optimizer import AdamPlanner, DerivativeFreePlanner
from core.utils.diffskill_utils import batch_pred_n, batch_pred, img_to_tensor, img_to_np
from core.diffskill.plan.project_func import generate_project_func, generate_batch_loss_func
from functools import partial


def optimize(n, args, agent, planner, tids, plan_step, project_func, tensor_dpc, tensor_goal_dpc, init_u=None):
    # n: number of seeds
    dimu = agent.dimu
    if init_u is None:
        init_u = agent.vae.sample_u(n * plan_step).reshape(n, plan_step * dimu)
    # Loss function:
    with torch.no_grad():  # Get latent code of initial and goal observation
        z_obs, t_obs = agent.vae.encode(tensor_dpc)
        u_obs = torch.cat([z_obs, t_obs], dim=1)
        z_goal, t_goal = agent.vae.encode(tensor_goal_dpc)
        u_goal = torch.cat([z_goal, t_goal], dim=1)
        tiled_u_obs, tiled_u_goal = u_obs.repeat([n, 1]), u_goal.repeat([n, 1])

    batch_loss_func = generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal)

    mask = torch.ones_like(init_u).int()
    if args.freeze_z:
        mask = mask.view(n, plan_step, dimu)
        mask[:, :, :-3] = 0  # Freeze the weights on z
        mask = mask.view(n, plan_step * dimu)
    sol_u, info = planner.optimize(init_u, batch_loss_func, project_func, mask)
    info['u_obs'], info['u_goal'] = u_obs.detach().cpu().numpy(), u_goal.detach().cpu().numpy()
    return sol_u, info


def optimize_rgbd(n, agent, planner, tids, plan_step, project_func, tensor_obs, tensor_goal_obs, init_u=None):
    if init_u is None:
        init_z = agent.vae.sample_latents(n * plan_step, agent.device).reshape(n, planner.dim)
    else:
        init_z = init_u

    # Loss function:
    with torch.no_grad():  # Get latent code of initial and goal observation
        _, z_obs, _ = agent.vae.encode(tensor_obs)
        _, z_goal_obs, _ = agent.vae.encode(tensor_goal_obs)
        tiled_z_obs, tiled_z_goal_obs = z_obs.repeat([n, 1]), z_goal_obs.repeat([n, 1])

    batch_loss_func = generate_batch_loss_func(agent, tids, tiled_z_obs, tiled_z_goal_obs)

    sol_z, info = planner.optimize(init_z, batch_loss_func, project_func)
    return sol_z, info


def plan(args, agent, plan_info, plan_step=2, profile=False, init_u=None, view_init=(140, -90)):
    """plan info should contain either (1) obs and goal_obs  (2) env, init_v and target_v, 'np_target_imgs"""
    args.dimu = agent.dimu
    if args.opt_mode == 'adam':
        planner = AdamPlanner(args, dim=plan_step * agent.dimu)
    elif args.opt_mode == 'dfp':
        planner = DerivativeFreePlanner(args, dim=plan_step * agent.dimu)
    else:
        raise NotImplementedError

    env = plan_info['env']
    state = env.reset([{'init_v': plan_info['init_v'], 'target_v': plan_info['target_v']}])[0]
    obs = env.render([{'mode': 'rgb', 'img_size': args.img_size}])[0]
    goal_obs = env.getattr('target_img', 0)
    goal_dpc = np.array(env.getattr('target_pc', 0))
    tensor_goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)[None]

    tensor_obs = img_to_tensor(np.array(obs[None]), agent.args.img_mode).to(agent.device, non_blocking=True)
    tensor_goal_obs = img_to_tensor(np.array(goal_obs[None]), agent.args.img_mode).to(agent.device, non_blocking=True)
    dpc = state[:3000].reshape(1000, 3)
    tensor_dpc = torch.FloatTensor(dpc).to(agent.device)[None]

    best_traj = None
    all_traj = []
    all_info = {}

    # Search over discrete variables
    search_tids_idxes = np.indices([args.num_tools] * plan_step).reshape(plan_step, -1).transpose()

    if profile:
        st_time = time.time()

    if args.input_mode != 'rgbd':
        t_min = torch.min(agent.vae.stat['mean'], dim=0, keepdim=True)[0]
        t_max = torch.max(agent.vae.stat['mean'], dim=0, keepdim=True)[0]
        project_func = generate_project_func(args, plan_step, t_min, t_max)  # Projection to the constraint set
    else:
        project_func = generate_project_func(args, plan_step)  # Projection to the constraint set

    if args.env_name == 'CutRearrangeSpread-v1' and plan_step == 6: 
        # For CRS-Twice, we give all the baselines the skill skeleton.
        search_tids_idxes = [np.array([0, 1, 2, 0, 1, 2])]

    for tids in search_tids_idxes:
        # Preparation
        if args.input_mode == 'rgbd':
            f = partial(optimize_rgbd, agent=agent, planner=planner, tids=tids, plan_step=plan_step, project_func=project_func,
                        tensor_obs=tensor_obs, tensor_goal_obs=tensor_goal_obs, init_u=init_u)
        else:
            f = partial(optimize, args=args, agent=agent, planner=planner, tids=tids, plan_step=plan_step, project_func=project_func,
                        tensor_dpc=tensor_dpc, tensor_goal_dpc=tensor_goal_dpc, init_u=init_u)
        sol_u, info = batch_pred_n(
            func=f,
            N=args.adam_sample,
            batch_size=args.plan_bs,
            collate_fn=planner.collate_fn)

        n, dimu = args.adam_sample, agent.dimu
        traj_score = info['traj_score'][-1]  #
        idxes = np.argsort(traj_score)[::-1][:3]  # Get the top 3 result

        # Visualize intermediate point clouds
        selected_u = sol_u.view(args.adam_sample, plan_step, dimu)[idxes.copy(), :, :].view(3 * plan_step, dimu)
        if args.input_mode == 'rgbd':
            np_mgoals = img_to_np(agent.vae.decode(selected_u)).reshape([3, plan_step, *obs.shape])
        else:
            np_mgoals = agent.vae.decode(selected_u, num_points=1000).detach().cpu().numpy()
            img_mgoals = [visualize_point_cloud_plt(pc, view_init=view_init) for pc in np_mgoals]
            dpc_img, goal_dpc_img = visualize_point_cloud_plt(dpc, view_init=view_init), visualize_point_cloud_plt(goal_dpc, view_init=view_init)

            img_all_plan = []
            for i in range(3):
                img_all_plan.append(dpc_img)
                for j in range(plan_step):
                    img_all_plan.append(img_mgoals[i * plan_step + j])
                img_all_plan.append(goal_dpc_img)
            img_shape = img_all_plan[0].shape
            img_all_plan = np.array(img_all_plan).reshape(3, plan_step + 2, *img_shape)

        if args.visualize_adam_pc:
            his_ts = info['his_t']  # 50 x 128 x planstep x 3
            his_zs = info['his_z']
            his_us = np.concatenate([his_zs, his_ts], axis=-1)

        sol_u = sol_u.view([n, plan_step, dimu])

        for rank, i in enumerate(idxes):  # Save the top 3 trajectories for each tid and also the best one
            if args.input_mode == 'rgbd':
                traj_img = [obs * 255.] + list(np_mgoals[rank, :] * 255.) + [goal_obs * 255.]
                info['u_obs'] = info['u_goal'] = None
            else:
                traj_img = img_all_plan[rank, :]
            traj_u = sol_u[i, :].detach().cpu().numpy()
            traj = {'traj_score': info['traj_score'][-1][i],
                    'traj_succ': info['traj_succ'][-1][:, i],
                    'pred_r': info['pred_r'][-1][i],
                    'tool': tids,
                    'zs': traj_u,
                    'traj_img': traj_img,
                    'init_v': plan_info['init_v'],
                    'target_v': plan_info['target_v'],
                    'u_obs': info['u_obs'],
                    'u_goal': info['u_goal']}
            opt_his = {'his_loss': info['losses'][:, i],
                       'his_traj_zlogl': info['traj_zlogl'][:, i],
                       'his_traj_score': info['traj_score'][:, i],
                       'his_traj_succ': info['traj_succ'][:, :, i],
                       'his_traj_pred_r': info['pred_r'][:, i],
                       }
            if args.visualize_adam_pc:
                if args.adam_iter < 50:
                    save_idx = np.arange(0, args.adam_iter)
                else:
                    save_idx = np.arange(0, args.adam_iter, args.adam_iter // 50)
                opt_his['his_traj_u'] = his_us[save_idx, i]  # his_us: [50, 128, plan_step, dimu]
                opt_his['dpc'] = dpc
                opt_his['goal_dpc'] = goal_dpc
            traj.update(**opt_his)
            if best_traj is None or traj['traj_score'] > best_traj['traj_score']:
                best_traj = traj
            all_traj.append(traj)

    if profile:
        all_info['plan_time'] = time.time() - st_time
        print('Plan time for the trajectory:', all_info['plan_time'])

    return best_traj, all_traj, all_info


def execute(env, agent, plan_traj, reset_primitive=False, save_name=None, demo=False, save_dir=None):
    state = env.reset([{'init_v': plan_traj['init_v'], 'target_v': plan_traj['target_v']}])[0]
    imgs, all_actions, all_resets = [], [], []
    for step in range(1, len(plan_traj['traj_img']) - 1):  # Skip the first observation since we only need the goals
        u_goal = torch.FloatTensor(plan_traj['zs'][step - 1, :]).to(agent.device)[None]
        tid = plan_traj['tool'][step - 1]
        print('tool id:', tid)
        primitive_state = env.getfunc('get_primitive_state', 0)

        for i in range(50):
            if i == 0 and agent.args.env_name == 'CutRearrangeSpread-v1' and agent.args.input_mode != 'rgbd':
                # Do matching of pcl, move the tool to the component closest to goal.
                # This is only done in CRS because of the way the environment is set up
                if step == 1:
                    dpc = state[:3000].reshape(-1, 1000, 3)
                    tensor_goal = agent.vae.decode(u_goal, num_points=1000)
                else:
                    tensor_goal = agent.vae.decode(u_goal, num_points=1000)
                    pcs_goal = tensor_goal.detach().cpu().numpy()
                    goal_label = dbscan_cluster(pcs_goal)
                    pcs_goal = decompose_pc(pcs_goal, goal_label, N=1000)
                    dpc_start = state[:3000].reshape(-1, 1000, 3)
                    label = dbscan_cluster(dpc_start)
                    pcs_obs = decompose_pc(dpc_start, label, N=1000)
                    dpc, goal_dpc, info = match_set_pcl(1, [pcs_obs], [pcs_goal], 1, 1, dist='chamfer',
                                                chamfer_downsample=200, verbose=False, eval=True, thr=5e-4)
                    tensor_goal = torch.FloatTensor(goal_dpc).to(agent.device)
                tool_reset_state = get_reset_tool_state(agent.args.env_name, dpc, tid)
                all_resets.append(tool_reset_state)
                env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': tool_reset_state}])
            obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]
            if agent.args.input_mode == 'rgbd':
                tensor_obs = img_to_tensor(obs[None], mode=agent.args.img_mode).cuda()
                target_img = agent.vae.decode(u_goal)
                action, done, _ = agent.act_rgbd(tensor_obs, target_img, tid)
            else:
                tensor_state = torch.FloatTensor(state[None]).to(agent.device)
                if 'pointnet' in agent.args.actor_arch:
                    action, done, _ = agent.act(tensor_state, tensor_goal, tid)
                else:
                    action, done, _ = agent.act_ugoal(tensor_state, u_goal, tid)
            action, done = action[0].detach().cpu().numpy(), done[0].detach().cpu().numpy()
            if np.round(done).astype(int) == 1 and agent.terminate_early and i > 0:
                break
            state, _, _, infos = env.step([action])
            info = infos[0]
            state = state[0]
            curr_img = obs
            write_number(curr_img, info['info_normalized_performance'], color=(1, 1, 1))
            imgs.append(curr_img)
            all_actions.append(action)
        for i in range(10):
            action = np.zeros(agent.args.action_dim)
            state, _, _, infos = env.step([action])
            info = infos[0]
            state = state[0]
            curr_img = obs
            write_number(curr_img, info['info_normalized_performance'], color=(1, 1, 1))
            imgs.append(curr_img)
            all_actions.append(action)

        # For ablation
        if agent.args.num_tools == 1:
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 0, 'reset_states': primitive_state}])
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 1, 'reset_states': primitive_state}])
        elif 'CutRearrange' not in agent.args.env_name:
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])
        else:
            tool_reset_state = get_reset_tool_state(agent.args.env_name, state[:3000].reshape(-1, 1000, 3), tid, back=True)
            env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': tool_reset_state}])
            actions, obses = [], []
        for obs, action in zip(obses, actions):
            if demo:
                write_number(obs, info['info_normalized_performance'], color=(1, 1, 1))
            imgs.append(obs)
            all_actions.append(action)
    score = info.get('info_normalized_performance', 0.)  # In case the agent terminates in the first step
    for i in range(20):
        imgs.append(curr_img)
        all_actions.append(np.zeros_like(action))

    if save_name is not None:
        save_numpy_as_gif(np.array(imgs) * 255, save_name)
        pkl_name = copy.copy(save_name)[:-4] + '.pkl'
        plan_traj['actions'] = np.array(all_actions)
        plan_traj['resets'] = np.array(all_resets)
        with open(pkl_name, 'wb') as f:
            pickle.dump(plan_traj, f)
    return imgs, score