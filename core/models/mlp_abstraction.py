import torch
import torch.nn as nn
from core.utils.diffskill_utils import weight_init

class FeasibilityPredictor(nn.Module):
    """ Predict feasibility"""

    def __init__(self, args, dimu):
        """ bin_succ: if true, use binary classification and otherwise use regression"""
        super().__init__()
        self.args = args
        self.bin_succ = args.bin_succ
        self.dimu = dimu
        hidden_dim = args.fea_latent_dim
        input_dim = dimu * 2
        if args.t_relative:
            input_dim -= 3
        if self.bin_succ:
            self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, 2))
            self.loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))
            self.loss = nn.MSELoss(reduction='none')
        self.softmax = nn.Softmax(dim=1) if args.bin_succ else nn.Softmax()
        self.apply(weight_init)

    def forward(self, obs, goal, eval):
        if self.args.t_relative:
            z_obs, t_obs = obs[:, :-3], obs[:, -3:]
            z_goal, t_goal = goal[:, :-3], goal[:, -3:]

            input = torch.cat([z_obs, z_goal, t_goal - t_obs], dim=1)
        else:
            input = torch.cat([obs, goal], dim=1)
        if self.args.bin_succ:
            pred = self.network(input)
            if eval:
                pred = self.softmax(pred)
                pred = pred[:, 1]  # Probability of success, i.e. being in category 1
        else:
            pred = self.network(input)[:, 0]  # Flatten last dim
            # Clamping makes the gradient zero. Use soft clipping instead
            if 'soft_clipping' in self.args.__dict__ and self.args.soft_clipping:
                pred = softclipping(pred, 0., 1.)
            else:
                if eval:
                    pred = torch.clamp(pred, 0., 1.)
        return pred


class RewardPredictor(nn.Module):
    """ Predict reward"""

    def __init__(self, args, dimu):
        super().__init__()
        self.args = args
        input_dim = dimu * 2
        if args.t_relative:
            input_dim -= 3
        hidden_dim = args.reward_latent_dim
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        self.loss = nn.MSELoss(reduction='none')
        self.apply(weight_init)

    def forward(self, obs, goal, **kwargs):
        if self.args.t_relative:
            z_obs, t_obs = obs[:, :-3], obs[:, -3:]
            z_goal, t_goal = goal[:, :-3], goal[:, -3:]

            input = torch.cat([z_obs, z_goal, t_goal - t_obs], dim=1)
        else:
            input = torch.cat([obs, goal], dim=1)
        pred = self.network(input)[:, 0]
        return pred

