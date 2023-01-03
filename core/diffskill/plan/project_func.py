import torch
from core.utils.diffskill_utils import traj_gaussian_logl, gaussian_logl, batch_pred


def generate_batch_loss_func(agent, tids, tiled_u_obs, tiled_u_goal):
    """
    :param tids: Tool to use at each step
    :param tiled_u_obs: tiled initial latent
    :param tiled_u_goal: tiled goal latent
    """

    def batch_loss_func(u):
        u = u.view(tiled_u_obs.shape[0], len(tids), -1)
        traj_succ = []
        for step, tid in enumerate(tids):
            if step == 0:
                curr_u = tiled_u_obs
            else:
                curr_u = u[:, step - 1]
            curr_goal = u[:, step]
            succ = batch_pred(agent.feas[tids[step]], {'obs': curr_u, 'goal': curr_goal, 'eval': True})
            traj_succ.append(succ[None, :])
        traj_succ = torch.cat(traj_succ, dim=0)  # [plan_step, num_plans]
        pred_r = batch_pred(agent.reward_predictor, {'obs': u[:, len(tids) - 1], 'goal': tiled_u_goal, 'eval': True})
        pred_r = torch.min(pred_r, torch.zeros_like(pred_r))  # score is negative emd, which should be less than zero
        traj_score = torch.max(traj_succ[0], torch.ones_like(traj_succ[0]) * 1e-2)
        for i in range(1, len(traj_succ)):
            traj_score = traj_score * torch.max(traj_succ[i], torch.ones_like(traj_succ[i]) * 1e-2)
        traj_score = traj_score * torch.exp(pred_r * 10.)
        total_loss = -traj_score

        # Add auxiliary losses for other steps as well. This will affect the CutRearrange
        for i in range(len(traj_succ)):
            total_loss += -0.01 * traj_succ[i]

        traj_zlogl = traj_gaussian_logl(u)
        return total_loss, {'traj_succ': traj_succ,
                            'pred_r': pred_r,
                            'traj_score': traj_score,
                            'traj_zlogl': traj_zlogl}

    return batch_loss_func


def generate_project_func(args, plan_step, t_min=None, t_max=None):
    if args.input_mode == 'rgbd':
        def project(z):
            copy_z = z.view(-1, plan_step, args.dimz)
            with torch.no_grad():
                z_logl = gaussian_logl(copy_z)
                projected_z = copy_z / torch.max(torch.sqrt((z_logl[:, :, None] / args.min_zlogl)), torch.ones(1, device=z.device))
            return projected_z.view(*z.shape).detach()

        return project if 'min_zlogl' in args.__dict__ else None

    else:
        def project(u):
            projected_u = u.view(-1, plan_step, args.dimu)
            with torch.no_grad():
                # projected_u[:, :, -3:] = torch.clip(projected_u[:, :, -3:], 0., 1.)  # Clip to zero and one. #TODO Use the actual boundary
                trans = projected_u[:, :, -3:]
                trans = torch.max(t_min.reshape(1, 1, 3), trans)
                trans = torch.min(t_max.reshape(1, 1, 3), trans)
                projected_u[:, :, -3:] = trans
            return projected_u.view(u.shape).detach()

        return project
