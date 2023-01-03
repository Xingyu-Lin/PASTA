import pickle
import copy
import numpy as np
import torch
import pytorch3d
from core.traj_opt.env_spec import get_reset_tool_state
from core.utils.diffskill_utils import write_number
from core.utils.pasta_utils import match_set_pcl
from core.utils.plb_utils import save_numpy_as_gif
from core.utils.pasta_utils import get_skill_in_out
from core.pasta.generate_dbscan_label import dbscan_cluster
from core.utils.pc_utils import decompose_pc, resample_pc
from core.utils.diffskill_utils import traj_gaussian_logl
import time

from core.pasta.plan.optimizer import AdamPlanner
from core.utils.diffskill_utils import batch_pred_n, batch_pred
from functools import partial
from core.pasta.plan.set_trajs import SetTrajs, struct_u_to_pc
from core.utils.open3d_utils import visualize_point_cloud_batch, visualize_point_cloud_plt
from core.utils.plb_utils import save_rgb, make_grid


def generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal, sin, sout):
    """
    :param tids: Tool to use at each step
    :param tiled_u_obs: tiled initial latent
    :param tiled_u_goal: tiled goal latent
    """
    set_trajs = SetTrajs(tids, sin, sout, tiled_u_obs, tiled_u_goal, agent.vae.sample_u)

    def batch_loss_func(all_u_opt, matches=None):  # Input is the current sub-goals to be optimized
        """ If matches is not None, then use it to predict reward to accelerate"""
        traj_succ, traj_heuristic = [], []
        all_u_opt, all_obs_goal = set_trajs.get_structured_u(all_u_opt)

        for step, (u_opt, obs_goal) in enumerate(zip(all_u_opt, all_obs_goal)):
            fea = batch_pred(agent.feas[tids[step]], {'obs_goal': obs_goal, 'eval': True})
            traj_succ.append(fea[None, :])
            if agent.args.coeff_heuristic > 0.:
                heuristic = batch_pred(agent.reward_predictor.predict_goal_nn, {'u_obs': u_opt, 'u_goal': tiled_u_goal})  # Smaller is better.
                traj_heuristic.append(heuristic[None, :])  # [1 x B]

        traj_succ = torch.cat(traj_succ, dim=0)  # [plan_step, num_plans]
        if agent.args.coeff_heuristic > 0.:
            traj_heuristic = torch.cat(traj_heuristic, dim=0) # [plan_step, num_plans]

        def collate_fn(list_ret):
            preds, matches = [], []
            for (pred, match) in list_ret:
                preds.append(pred)
                matches.append(match)
            return torch.cat(preds, dim=0), np.vstack(matches)
        def collate_fn_gt_r(preds):
            return torch.cat(preds, dim=0)

        if agent.args.gt_reward:
            pred_r = batch_pred(agent.get_gt_reward, {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal}, batch_size=200, 
                                collate_fn=collate_fn_gt_r)
        else:
            if matches is not None:
                pred_r = batch_pred(agent.reward_predictor.predict_match, {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal, 'match': matches})
            else:
                pred_r, matches = batch_pred(agent.reward_predictor.predict_array, {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal, 'ret_match': True},
                                            collate_fn=collate_fn)

        pred_r = - torch.sum(pred_r, dim=1)
        pred_r = torch.min(pred_r, torch.zeros_like(pred_r))  # score is negative emd, which should be less than zero
        traj_score = torch.max(traj_succ[0], torch.ones_like(traj_succ[0]) * 1e-2)
        for i in range(1, len(traj_succ)):
            traj_score = traj_score * torch.max(traj_succ[i], torch.ones_like(traj_succ[i]) * 1e-2)
        rew_coeff = 100. if 'CutRearrangeSpread' in agent.args.env_name else 10.
        traj_score = traj_score * torch.exp(pred_r * rew_coeff)
        total_loss = -traj_score

        # Add auxiliary losses for other steps as well. This will affect the CutRearrange
        heuristic_loss = 0.
        for i in range(len(traj_succ)):
            total_loss += -0.01 * traj_succ[i]
            if agent.args.coeff_heuristic > 0.:
                heuristic_loss += -agent.args.coeff_heuristic * traj_heuristic[i]

        if agent.args.coeff_heuristic > 0.:
            print('total_loss: {}, heuristic_loss: {}'.format(torch.sum(total_loss), torch.sum(heuristic_loss)))
            total_loss += heuristic_loss
        return total_loss, {'traj_succ': traj_succ,
                            'pred_r': pred_r,
                            'traj_score': traj_score,
                            'traj_zlogl': 0.}, matches

    return batch_loss_func, set_trajs


def generate_project_func(args, plan_step, t_min=None, t_max=None):
    def project(u):
        projected_u = u
        with torch.no_grad():
            trans = projected_u[:, -3:]
            trans = torch.max(t_min.reshape(1, 3), trans)
            trans = torch.min(t_max.reshape(1, 3), trans)
            projected_u[:, -3:] = trans
        return projected_u.view(u.shape).detach()

    return project


def optimize(n, args, agent, planner, tids, project_func, tensor_dpc, tensor_goal_dpc, sin, sout, init_u=None):
    # n: number of seeds
    assert init_u is None

    # Loss function:
    with torch.no_grad():  # Get latent code of initial and goal observation
        u_obs, u_goal = agent.vae.encode_u(tensor_dpc)[None], agent.vae.encode_u(tensor_goal_dpc)[None]
        tiled_u_obs, tiled_u_goal = u_obs.repeat([n, 1, 1]), u_goal.repeat([n, 1, 1])
    # Enumerate and build the graph
    batch_loss_func, set_trajs = generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal, sin, sout)

    assert not args.freeze_z
    sol_u, info = planner.optimize(set_trajs.all_u_opt, batch_loss_func, project_func, None)
    if args.visualize_adam_pc:
        set_trajs.set_sol(sol_u, info['his_u'])
    else:
        set_trajs.set_sol(sol_u)
    info['u_obs'], info['u_goal'] = u_obs.detach().cpu().numpy(), u_goal.detach().cpu().numpy()
    return set_trajs, info


def plan(args, agent, plan_info, plan_step=2, profile=False, init_u=None, view_init=(140, -90)):
    """plan info should contain either (1) obs and goal_obs  (2) env, init_v and target_v, 'np_target_imgs"""
    args.dimu = agent.dimu
    planner = AdamPlanner(args, dim=plan_step * agent.dimu)
    env = plan_info['env']
    state = env.reset([{'init_v': plan_info['init_v'], 'target_v': plan_info['target_v']}])[0]
    goal_dpc = np.array(env.getattr('target_pc', 0))

    dpc = state[:3000].reshape(1000, 3)

    labels = dbscan_cluster(np.array([dpc, goal_dpc]))
    set_pcs = decompose_pc(dpc, labels[0], N=1000)
    set_gpcs = decompose_pc(goal_dpc, labels[1], N=1000)
    tensor_set_pcs = torch.FloatTensor(set_pcs).to(agent.device)
    tensor_set_gpcs = torch.FloatTensor(set_gpcs).to(agent.device)
    diff = len(tensor_set_gpcs) - len(tensor_set_pcs)

    best_traj, all_traj, all_info = None, [], {}
    # Search over discrete variables
    search_tids_idxes = np.indices([args.num_tools] * plan_step).reshape(plan_step, -1).transpose()

    def filter_tids(search_tids, sin, sout, diff):
        return [tids for tids in search_tids if sum(sout[tids]) - sum(sin[tids]) == diff]

    sin, sout = get_skill_in_out(args.env_name, filter_set=args.filter_set)
    search_tids_idxes = filter_tids(search_tids_idxes, sin, sout, diff)

    if plan_step == 6:
        search_tids_idxes = [np.array([0, 1, 2, 0, 1, 2])]
    print('Filtered tids to be searched: ', search_tids_idxes)

    if profile:
        st_time = time.time()

    t_min = torch.min(agent.vae.stat['mean'], dim=0, keepdim=True)[0]
    t_max = torch.max(agent.vae.stat['mean'], dim=0, keepdim=True)[0]

    project_func = generate_project_func(args, plan_step, t_min, t_max)  # Projection to the constraint set
    for tids in search_tids_idxes:
        set_trajs, info = batch_pred_n(
            func=partial(optimize, args=args, agent=agent, planner=planner, tids=tids, project_func=project_func,
                         tensor_dpc=tensor_set_pcs, tensor_goal_dpc=tensor_set_gpcs, init_u=init_u, sin=sin, sout=sout),
            N=args.adam_sample,
            batch_size=args.plan_bs,
            collate_fn=planner.collate_fn)

        traj_score = info['traj_score'][-1]
        topk = 4 if args.adam_sample > 4 else args.adam_sample
        idxes = np.argsort(traj_score)[::-1][:topk]  # Get the top 3 result

        # Visualize intermediate point clouds
        select_struct_u = set_trajs.select(idxes)  # [H] x B x tot x D
        pcs, list_pc = struct_u_to_pc(agent, select_struct_u)  # B x plan_step, # [H] x B x tot x D
        img_mgoals = visualize_point_cloud_batch(pcs.reshape(-1, 1000, 3))
        dpc_img, goal_dpc_img = visualize_point_cloud_plt(dpc, view_init=view_init), visualize_point_cloud_plt(goal_dpc, view_init=view_init)

        img_all_plan = []
        for i in range(topk):
            img_all_plan.append(dpc_img)
            for j in range(plan_step):
                img_all_plan.append(img_mgoals[i * plan_step + j])
            img_all_plan.append(goal_dpc_img)
        img_shape = img_all_plan[0].shape
        img_all_plan = np.array(img_all_plan).reshape(topk, plan_step + 2, *img_shape)
        # img_all_plan = np.array(img_all_plan).reshape(3 * (plan_step + 2), *img_shape)
        # img_all_plan = make_grid(img_all_plan)
        # save_rgb('./data/debug/set_{}.png'.format(tids), img_all_plan)

        if args.visualize_adam_pc:
            select_struct_u_his = set_trajs.select_his(idxes)    # [H] x T x B x tot x D

        for rank, i in enumerate(idxes):  # Save the top 3 trajectories for each tid and also the best one
            traj_img = img_all_plan[rank, :]
            # traj_u = sol_u[i, :].detach().cpu().numpy()
            traj = {'traj_score': info['traj_score'][-1][i],
                    'traj_succ': info['traj_succ'][-1][:, i],
                    'pred_r': info['pred_r'][-1][i],
                    'tool': tids,
                    'traj_img': traj_img,
                    'init_v': plan_info['init_v'],
                    'target_v': plan_info['target_v'],
                    'us': [select_struct_u[t][rank].detach().cpu().numpy() for t in range(len(select_struct_u))],
                    'masks_in': np.array([set_trajs.masks_in[t][i] for t in range(len(set_trajs.masks_in))]),
                    'u_obs': info['u_obs'],
                    'u_goal': info['u_goal'],
                    'pcs': [dpc.reshape(-1, 1000, 3)] + [list_pc[i][rank] for i in range(len(list_pc))] + [goal_dpc.reshape(-1, 1000, 3)]}
            opt_his = {'his_loss': info['losses'][:, i],
                       # 'his_traj_zlogl': info['traj_zlogl'][:, i],
                       'his_traj_score': info['traj_score'][:, i],
                       'his_traj_succ': info['traj_succ'][:, :, i],
                       'his_traj_pred_r': info['pred_r'][:, i]
                       }

            if args.visualize_adam_pc:
                if args.adam_iter < 50:
                    save_idx = np.arange(0, args.adam_iter)
                else:
                    save_idx = np.arange(0, args.adam_iter, args.adam_iter // 50)
                opt_his['struct_u_his'] = [u_his[save_idx, rank, :, :] for u_his in select_struct_u_his]   # [H] x 50 x tot x D
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


def execute(env, agent, plan_traj, reset_primitive=False, save_name=None, demo=False, save_dir=''):
    state = env.reset([{'init_v': plan_traj['init_v'], 'target_v': plan_traj['target_v']}])[0]
    imgs, all_actions, all_resets = [], [], []
    for step in range(1, len(plan_traj['traj_img']) - 1):  # Skip the first observation since we only need the goals
        tid = plan_traj['tool'][step - 1]
        print('tool id:', tid)
        n_in, n_out = agent.skill_def[tid]['in'], agent.skill_def[tid]['out']
        primitive_state = env.getfunc('get_primitive_state', 0)
        # Getting goal
        pcs_goal = []
        u_goals = plan_traj['us'][step - 1]
        for u_goal in u_goals:
            pcs = agent.vae.decode(torch.FloatTensor(u_goal[None]).to(agent.device), 1000).squeeze(0)
            pcs_goal.append(pcs.detach().cpu().numpy())

        # if not agent.args.filter_set:
        exec_flag = True
        dpc_start = state[:3000].reshape(-1, 1000, 3)
        goal_dpc = np.vstack(pcs_goal)[None]
        goal_dpc = torch.FloatTensor(goal_dpc).to(agent.device)
        filtered_obs_idx = np.arange(1000)
        obs_dpc = torch.FloatTensor(dpc_start).to(agent.device)
        dist, _ = pytorch3d.loss.chamfer_distance(obs_dpc, goal_dpc)
        if dist.item() < agent.args.exec_threshold:
            exec_flag = False
        print("exec flag: ", exec_flag)

        if exec_flag and 'masks_in' in plan_traj.keys():
            mask_in = plan_traj['masks_in'][step - 1]
            dpc_start = state[:3000].reshape(-1, 1000, 3)
            label = dbscan_cluster(dpc_start)
            pcs_obs = decompose_pc(dpc_start, label, N=1000)
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
        elif exec_flag:
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
                if step == 1:
                    break
                min_samples += 1
        pcl_imgs = []
        for i in range(50):
            dpc = state[:3000].reshape(-1, 1000, 3)
            dpc = dpc[:, filtered_obs_idx, :]
            dpc = resample_pc(dpc[0], 1000)[None]
            obs = env.render([{'mode': 'rgb', 'img_size': agent.args.img_size}])[0]
            if exec_flag:
                if i == 0 and 'CutRearrangeSpread' in agent.args.env_name:
                    tool_reset_state = get_reset_tool_state(agent.args.env_name, dpc, tid)
                    all_resets.append(tool_reset_state)
                    env.getfunc(f'taichi_env.primitives[{tid}].set_state', 0, list_kwargs=[{'f': 0, 'state': tool_reset_state}])
                state = torch.FloatTensor(state).to(agent.device)[None]
                dpc = torch.FloatTensor(dpc).to(agent.device)
                action, done, img = agent.act(state, dpc, goal_dpc, tid)
                if img is not None:
                    pcl_imgs.append(img)
                action, done = action[0].detach().cpu().numpy(), done[0].detach().cpu().numpy()
            else:
                action, done = np.zeros((agent.args.action_dim)), 1.
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

        import os
        if len(pcl_imgs) > 0:
            init_v, target_v = plan_traj['init_v'], plan_traj['target_v']
            save_numpy_as_gif(np.array(pcl_imgs), os.path.join(save_dir, f'init{init_v}_target{target_v}-step{step}.gif'))
        # For ablation
        if agent.args.num_tools == 1:
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 0, 'reset_states': primitive_state}])
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 1, 'reset_states': primitive_state}])
        elif 'CutRearrange' not in agent.args.env_name:
            _, actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])
        else:
            tool_reset_state = get_reset_tool_state(agent.args.env_name, dpc, tid, back=True)
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
