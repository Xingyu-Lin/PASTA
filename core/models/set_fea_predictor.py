import torch
import torch.nn as nn
from core.utils.diffskill_utils import weight_init
from core.utils.diffskill_utils import softclipping

class SetFeasibilityPredictor(nn.Module):
    """ Predict feasibility"""

    def __init__(self, args, input_dim):
        """ bin_succ: if true, use binary classification and otherwise use regression"""
        super().__init__()
        self.args = args
        self.bin_succ = args.bin_succ
        hidden_dim = args.fea_latent_dim
        self.arch = args.fea_arch if 'fea_arch' in args.__dict__ else 'v0'
        self.dimz, self.dimu = args.dimz, args.dimz + 3
        if self.arch == 'v0' or (self.arch == 'v3' and input_dim // self.dimu == 2):
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
        elif self.arch == 'v1':  # zs -> score, ts -> score, combine.
            n_tot = input_dim // self.dimu
            hidden_dim = 256
            assert input_dim % self.dimu == 0
            self.z_network = nn.Sequential(nn.Linear(self.dimz * n_tot, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 1),
                                           nn.Sigmoid())
            self.t_network = nn.Sequential(nn.Linear(3 * n_tot, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 1),
                                           nn.Sigmoid())
            self.loss = nn.MSELoss(reduction='none')
        elif self.arch == 'v2':  # Deeper network
            if args.t_relative:
                input_dim -= 3
            if self.bin_succ:
                self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
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
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, 1))
                self.loss = nn.MSELoss(reduction='none')
        elif self.arch == 'v3' and input_dim // self.dimu == 3:
            assert input_dim % self.dimu == 0 and not self.bin_succ and not args.t_relative
            hidden_dim = 256
            self.net1 = nn.Sequential(nn.Linear(self.dimu, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU())
            self.net2 = nn.Sequential(nn.Linear(self.dimu, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU())
            self.net3 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
            self.loss = nn.MSELoss(reduction='none')
        elif self.arch == 'v4':
            hidden_dim = 256
            self.net1 = nn.Sequential(nn.Linear(self.dimu, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU())
            self.net2 = nn.Sequential(nn.Linear(self.dimu, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU())
            self.net3 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
            self.loss = nn.MSELoss(reduction='none')


        self.softmax = nn.Softmax(dim=1) if args.bin_succ else nn.Softmax()
        self.apply(weight_init)

    def forward(self, obs_goal, eval):
        if self.args.fea_center:
            B = obs_goal.shape[0]
            obs_goal = obs_goal.view(B, -1, self.dimu)
            zeros = torch.zeros_like(obs_goal).to(obs_goal.device)
            t_mean = torch.mean(obs_goal[:, :, -3:], dim=1).view(B, 1, -1)
            zeros[:, :, -3:] -= t_mean
            obs_goal = obs_goal + zeros
            obs_goal = obs_goal.view(B, -1)

        if self.arch == 'v3' and obs_goal.shape[-1] // self.dimu == 3:
            B = obs_goal.shape[0]
            obs_goal = obs_goal.view(B, -1, self.dimu)
            u_obs, u_goal1, u_goal2 = obs_goal[:, 0, :], obs_goal[:, 1, :], obs_goal[:, 2, :]
            latent1 = self.net1(u_obs)
            latent2 = torch.maximum(self.net2(u_goal1), self.net2(u_goal2))
            pred = self.net3(torch.cat([latent1, latent2], dim=-1))[:, 0]
            # Clamping makes the gradient zero. Use soft clipping instead
            if 'soft_clipping' in self.args.__dict__ and self.args.soft_clipping:
                pred = softclipping(pred, 0., 1.)
            else:
                if eval:
                    pred = torch.clamp(pred, 0., 1.)
            return pred

        elif self.arch in ['v0', 'v2', 'v3']:
            if self.args.bin_succ:
                pred = self.network(obs_goal)
                if eval:
                    pred = self.softmax(pred)
                    pred = pred[:, 1]  # Probability of success, i.e. being in category 1
            else:
                pred = self.network(obs_goal)[:, 0]  # Flatten last dim
                # Clamping makes the gradient zero. Use soft clipping instead
                if 'soft_clipping' in self.args.__dict__ and self.args.soft_clipping:
                    pred = softclipping(pred, 0., 1.)
                else:
                    if eval:
                        pred = torch.clamp(pred, 0., 1.)
            return pred
        elif self.arch == 'v4':
            B = obs_goal.shape[0]
            obs_goal = obs_goal.view(B, -1, self.dimu)
            u_obs, u_goal = obs_goal[:, :-2, :], obs_goal[:, -2:, :]   # assuming it's 1->2 or 2->2
            latent1 = self.net1(u_obs)
            latent1, _ = torch.max(latent1, dim=1)
            latent2 = self.net2(u_goal)
            latent2, _ = torch.max(latent2, dim=1)
            pred = self.net3(torch.cat([latent1, latent2], dim=-1))[:, 0]
            # Clamping makes the gradient zero. Use soft clipping instead
            if 'soft_clipping' in self.args.__dict__ and self.args.soft_clipping:
                pred = softclipping(pred, 0., 1.)
            else:
                if eval:
                    pred = torch.clamp(pred, 0., 1.)
            return pred
        else:
            B = obs_goal.shape[0]
            obs_goal = obs_goal.view(B, -1, self.dimu)
            z_obs_goal = obs_goal[:, :, :-3].reshape(B, -1)
            t_obs_goal = obs_goal[:, :, -3:].reshape(B, -1)
            pred_z, pred_t = self.z_network(z_obs_goal), self.t_network(t_obs_goal)
            return (pred_z * pred_t)[:, 0]