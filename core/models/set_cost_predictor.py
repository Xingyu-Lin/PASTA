import torch
import torch.nn as nn
import numpy as np
from core.utils.diffskill_utils import weight_init
from scipy.optimize import linear_sum_assignment


class SetCostPredictor(nn.Module):
    """ Predict reward (Actually it's the dist)"""

    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        hidden_dim = args.reward_latent_dim
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        if self.args.rew_loss == 'mse':
            self.loss = nn.MSELoss(reduction='none')
        else:
            self.loss = nn.HuberLoss(reduction='none', delta=args.rew_huberd)
        self.apply(weight_init)

    def forward(self, u_obs_goal, **kwargs):
        if self.args.rew_center:
            B = u_obs_goal.shape[0]
            u_obs_goal = u_obs_goal.view(B, -1, self.args.dimz + 3)
            zeros = torch.zeros_like(u_obs_goal).to(u_obs_goal.device)
            t_mean = torch.mean(u_obs_goal[:, :, -3:], dim=1).view(B, 1, -1)
            zeros[:, :, -3:] -= t_mean
            u_obs_goal = u_obs_goal + zeros
            u_obs_goal = u_obs_goal.view(B, -1)
        pred = self.network(u_obs_goal)[:, 0]
        return pred

    def predict_list(self, list_u_obs, list_u_goal):
        raise NotImplementedError

    def predict_array(self, u_obs, u_goal, ret_match):
        # u_obs: [B x Ko x Dim_u]
        # u_goal: [B x Kg x Dim_u]
        B, Ko, D = u_obs.shape
        B, Kg, D = u_goal.shape
        u_obs_paired = u_obs.view(B, Ko, 1, D).repeat(1, 1, Kg, 1).view(B * Ko * Kg, D)
        u_goal_paired = u_goal.view(B, 1, Kg, D).repeat(1, Ko, 1, 1).view(B * Ko * Kg, D)
        dist_tensor = self.forward(torch.cat([u_obs_paired, u_goal_paired], dim=1)).view(B, Ko, Kg)
        dist = dist_tensor.detach().cpu().numpy()
        selected_dist, matches = [], []
        for i in range(B):
            row_ind, col_ind = linear_sum_assignment(dist[i])
            assert row_ind[0] == 0 and row_ind[-1] == len(dist[i]) - 1
            matches.append(col_ind)
            selected_dist.append(dist_tensor[i][row_ind, col_ind])
        selected_dist = torch.stack(selected_dist, dim=0)
        if ret_match:
            return selected_dist, np.array(matches)
        else:
            return selected_dist

    def predict(self, u_obs, u_goal, ret_match=False):
        if isinstance(u_obs, list):  # Batch predict with different cardinality
            return self.predict_list(u_obs, u_goal)
        elif isinstance(u_obs, torch.Tensor):
            return self.predict_array(u_obs, u_goal, ret_match)
        else:
            raise NotImplementedError

    def predict_match(self, u_obs, u_goal, match):
        """ u_obs: B x Ko x D
            u_goal: B x Kg x D
            match: B x Ko
        """
        B, Ko, D = u_obs.shape
        B, Kg, D = u_goal.shape
        match_idx = (match + np.arange(B).reshape(B, 1) * Kg).reshape(B * Ko, 1)     # this will error if B = 1
        u_goal_match = u_goal.view(B * Kg, D)[match_idx].view(B, Ko, D)

        # for i in range(u_obs.shape[0]):
        #     vec = torch.cat([u_obs[i][None], u_goal[i][match[i]][None]], dim=2)
        #     u_obs_goal.append(vec)
        # u_obs_goal = torch.cat(u_obs_goal, dim=0).view(B * K, D * 2)
        u_obs_goal = torch.cat([u_obs, u_goal_match], dim=-1).view(B * Ko, D * 2)
        pred_r = self.forward(u_obs_goal)
        return pred_r.view(B, Ko)

    def predict_goal_nn(self, u_obs, u_goal):
        """ Predict a heuristic reward between an observation and a goal set with different cardinalities.
            For each component in the goal, find its match in the observation with the highest reward and add them. """
        # u_obs, u_goal: [B x K x Dim_u]
        B, K_o, D = u_obs.shape
        B, K_g, D = u_goal.shape
        u_obs_paired = u_obs.view(B, K_o, 1, D).repeat(1, 1, K_g, 1).view(B * K_o * K_g, D)
        u_goal_paired = u_goal.view(B, 1, K_g, D).repeat(1, K_o, 1, 1).view(B * K_o * K_g, D)
        dist_tensor = self.forward(torch.cat([u_obs_paired, u_goal_paired], dim=1)).view(B, K_o, K_g)
        goal_nn_dist, _ = torch.min(dist_tensor, dim=1)
        return torch.sum(goal_nn_dist, dim=1)
