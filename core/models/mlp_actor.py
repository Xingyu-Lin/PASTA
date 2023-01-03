import torch
import torch.nn as nn
from core.utils.diffskill_utils import weight_init


class ActorV0(nn.Module):
    """ V0: Base"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        hidden_dim = args.actor_latent_dim
        self.action_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, state_obs_goal):
        return self.action_mlp(state_obs_goal), self.done_mlp(state_obs_goal)


class ActorV1(nn.Module):
    """ V1: Pre-process the tool state"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.dimu, self.dimtool = args.dimu, args.dimtool
        hidden_dim = args.actor_latent_dim
        self.state_encoder = nn.Sequential(nn.Linear(self.dimtool, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 16))
        self.action_mlp = nn.Sequential(nn.Linear(self.dimu * 2 + 16, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(self.dimu * 2 + 16, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, state_obs_goal):
        dimu, dimtool = self.args.dimu, self.args.dimtool
        u_obs, tool_state, u_goal = state_obs_goal[:, :dimu], state_obs_goal[:, dimu:dimu + dimtool], state_obs_goal[:, -dimu:]
        encode_tool = self.state_encoder(tool_state)
        concat = torch.cat([u_obs, encode_tool, u_goal], dim=1)
        return self.action_mlp(concat), self.done_mlp(concat)


class ActorV2(nn.Module):
    """ V2: Add tanh """

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        hidden_dim = args.actor_latent_dim
        self.action_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim),
                                        nn.Tanh())
        self.done_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, state_obs_goal):
        return self.action_mlp(state_obs_goal), self.done_mlp(state_obs_goal)


class ActorV3(nn.Module):
    """ V3: Shallower"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        hidden_dim = args.actor_latent_dim
        self.action_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, state_obs_goal):
        return self.action_mlp(state_obs_goal), self.done_mlp(state_obs_goal)

class ActorV4(nn.Module):
    """ V4: Maxpool"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.arch = args.actor_arch
        self.input_dim = input_dim
        hidden_dim = args.actor_latent_dim
        self.dimz, self.dimu = args.dimz, args.dimz + 3
        self.net1 = nn.Sequential(nn.Linear(self.dimu, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU())
        self.action_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, obs_goal):
        B = obs_goal.shape[0]
        obs_goal = obs_goal.view(B, -1, self.dimu)
        latents = self.net1(obs_goal)
        out, _ = torch.max(latents, dim=1)
        return self.action_mlp(out), self.done_mlp(out)