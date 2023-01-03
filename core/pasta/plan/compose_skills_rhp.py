import pickle
import copy
import numpy as np
import torch
from core.utils.pasta_utils import get_skill_in_out
from core.pasta.generate_dbscan_label import dbscan_cluster
from core.utils.pc_utils import decompose_pc
import time

from core.pasta.plan.optimizer import AdamPlanner
from core.utils.diffskill_utils import batch_pred_n, batch_pred
from functools import partial
from core.pasta.plan.set_trajs import SetTrajs, struct_u_to_pc
from core.utils.open3d_utils import visualize_point_cloud_batch, visualize_point_cloud_plt


def generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal, sin, sout):
    """
    :param tids: Tool to use at each step
    :param tiled_u_obs: tiled initial latent
    :param tiled_u_goal: tiled goal latent
    """
    set_trajs = SetTrajs(tids, sin, sout, tiled_u_obs, tiled_u_goal, agent.vae.sample_u)

    def batch_loss_func(all_u_opt, matches=None):  # Input is the current sub-goals to be optimized
        """ If matches is not None, then use it to predict reward to accelerate"""
        traj_succ = []
        all_u_opt, all_obs_goal = set_trajs.get_structured_u(all_u_opt)

        for step, (u_opt, obs_goal) in enumerate(zip(all_u_opt, all_obs_goal)):
            fea = batch_pred(agent.feas[tids[step]], {'obs_goal': obs_goal, 'eval': True})
            traj_succ.append(fea[None, :])

        traj_succ = torch.cat(traj_succ, dim=0)  # [plan_step, num_plans]

        def collate_fn(list_ret):
            preds, matches = [], []
            for (pred, match) in list_ret:
                preds.append(pred)
                matches.append(match)
            return torch.cat(preds, dim=0), np.vstack(matches)

        if matches is not None:
            pred_r = batch_pred(agent.reward_predictor.predict_match, {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal, 'match': matches})
        else:
            pred_r, matches = batch_pred(agent.reward_predictor.predict_array,
                                         {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal, 'ret_match': True},
                                         collate_fn=collate_fn)
        pred_r = - torch.sum(pred_r, dim=1)
        pred_r = torch.min(pred_r, torch.zeros_like(pred_r))  # score is negative emd, which should be less than zero
        r = 100 * pred_r
        # else:
        #     heuristic_r = batch_pred(agent.reward_predictor.predict_goal_nn, {'u_obs': all_u_opt[-1], 'u_goal': tiled_u_goal})  # Smaller is better.
        #     r = - 0.1 * heuristic_r

        traj_score = torch.max(traj_succ[0], torch.ones_like(traj_succ[0]) * 1e-2)
        for i in range(1, len(traj_succ)):
            traj_score = traj_score * torch.max(traj_succ[i], torch.ones_like(traj_succ[i]) * 1e-2)
        traj_score = traj_score * torch.exp(r * 1.)
        total_loss = -traj_score

        # Add auxiliary losses for other steps as well. This will affect the CutRearrange
        for i in range(len(traj_succ)):
            total_loss += -0.01 * traj_succ[i]

        return total_loss, {'traj_succ': traj_succ,
                            'pred_r': r,
                            'traj_score': traj_score,
                            'traj_zlogl': 0.}, matches

    return batch_loss_func, set_trajs


def generate_project_func(args, plan_step, t_min=None, t_max=None):
    def project(u):
        projected_u = u
        with torch.no_grad():
            # projected_u[:, :, -3:] = torch.clip(projected_u[:, :, -3:], 0., 1.)  # Clip to zero and one. #TODO Use the actual boundary
            trans = projected_u[:, -3:]
            trans = torch.max(t_min.reshape(1, 3), trans)
            trans = torch.min(t_max.reshape(1, 3), trans)
            projected_u[:, -3:] = trans
        return projected_u.view(u.shape).detach()

    return project


def optimize(n, args, agent, planner, tids, project_func, tensor_dpc, tensor_goal_dpc, sin, sout, init_u=None, t=0):
    # n: number of seeds
    assert init_u is None

    # Loss function:
    if t == 0:
        with torch.no_grad():  # Get latent code of initial and goal observation
            u_obs, u_goal = agent.vae.encode_u(tensor_dpc)[None], agent.vae.encode_u(tensor_goal_dpc)[None]
            tiled_u_obs, tiled_u_goal = u_obs.repeat([n, 1, 1]), u_goal.repeat([n, 1, 1])
    else:
        with torch.no_grad():  # Get latent code of initial and goal observation
            u_goal = agent.vae.encode_u(tensor_goal_dpc)[None]
            u_obs = tensor_dpc
            tiled_u_obs, tiled_u_goal = u_obs.repeat([n, 1, 1]), u_goal.repeat([n, 1, 1])

    # Enumerate and build the graph
    batch_loss_func, set_trajs = generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal, sin, sout)

    assert not args.freeze_z
    sol_u, info = planner.optimize(set_trajs.all_u_opt, batch_loss_func, project_func, None)
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

    sin, sout = get_skill_in_out(args.env_name)
    search_tids_idxes = [np.array([0, 1, 2, 0, 1, 2])]
    print('Filtered tids to be searched: ', search_tids_idxes)

    if profile:
        st_time = time.time()

    t_min = torch.min(agent.vae.stat['mean'], dim=0, keepdim=True)[0]
    t_max = torch.max(agent.vae.stat['mean'], dim=0, keepdim=True)[0]

    project_func = generate_project_func(args, plan_step, t_min, t_max)  # Projection to the constraint set
    tids = search_tids_idxes[0]
    T, H = len(tids), args.rhp_horizon
    args.plan_T = T

    curr_sols, curr_masks_in = [], []
    for t in range(0, T):
        if t == 0:
            obs = tensor_set_pcs
        else:
            obs = curr_sols[-1]
        set_trajs, info = batch_pred_n(
            func=partial(optimize, args=args, agent=agent, planner=planner, tids=tids[t:t + H], project_func=project_func,
                         tensor_dpc=obs, tensor_goal_dpc=tensor_set_gpcs, init_u=init_u, sin=sin, sout=sout, t=t),
            N=args.adam_sample,
            batch_size=args.plan_bs,
            collate_fn=planner.collate_fn)  # TODO set_trajs only collate a part of the structure. See collate_fn !!!
        traj_score = info['traj_score'][-1]
        idx = np.argsort(traj_score)[::-1][0]
        select_struct_u = set_trajs.select([idx])  # [H] x B x tot x D
        curr_struct_u = curr_sols + select_struct_u  # t + H
        curr_sols.append(select_struct_u[0])
        curr_masks_in.append(set_trajs.masks_in[0][idx])

        # log intermediate results
        pcs, list_pc = struct_u_to_pc(agent, curr_struct_u)  # B x plan_step, # [H] x B x tot x D
        img_mgoals = visualize_point_cloud_batch(pcs.reshape(-1, 1000, 3))

        def pad_array(a, T, val):
            if len(a) < T:
                M = T - len(a)
                pad_a = np.ones_like(a[[0] * M]) * val
                a = np.vstack([a, pad_a])
            return a

        img_mgoals = pad_array(np.array(img_mgoals), T, val=255.)
        dpc_img, goal_dpc_img = visualize_point_cloud_plt(dpc, view_init=view_init), visualize_point_cloud_plt(goal_dpc, view_init=view_init)

        img_all_plan = np.array([dpc_img, *img_mgoals, goal_dpc_img])
        traj_succ = list(info['traj_succ'][-1][:, idx])
        traj_succ = [np.nan] * t + traj_succ + [np.nan] * max(T - H - t, 0)
        traj = {'traj_score': info['traj_score'][-1][idx],
                'traj_succ': traj_succ,
                'pred_r': info['pred_r'][-1][idx],
                'tool': tids,
                'traj_img': img_all_plan,
                'init_v': plan_info['init_v'],
                'target_v': plan_info['target_v'],
                'us': [curr_struct_u[i][0].detach().cpu().numpy() for i in range(len(curr_struct_u))],
                'u_obs': info['u_obs'],
                'u_goal': info['u_goal'],
                'masks_in': np.array(curr_masks_in),  # H x 1
                'pcs': [dpc.reshape(-1, 1000, 3)] + [list_pc[i][0] for i in range(len(list_pc))] + [goal_dpc.reshape(-1, 1000, 3)]}
        opt_his = {'his_loss': info['losses'][:, idx],
                   'his_traj_score': info['traj_score'][:, idx],
                   'his_traj_succ': info['traj_succ'][:, :, idx],
                   'his_traj_pred_r': info['pred_r'][:, idx]}
        traj.update(**opt_his)
        all_traj.append(traj)

        if profile:
            all_info['plan_time'] = time.time() - st_time
            print('Plan time for the trajectory:', all_info['plan_time'])
    best_traj = all_traj[-1]
    return best_traj, all_traj, all_info
