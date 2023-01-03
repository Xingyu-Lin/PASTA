import torch
import torch.nn as nn
from core.utils.diffskill_utils import weight_init
from core.models.pointnet_encoder import PointNetEncoder, PointNetEncoderCat

class PointActorCat(nn.Module):
    """ PointCloud based actor with tool state concatenation"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoderCat(2)
        hidden_dim = args.actor_latent_dim
        self.dimu, self.dimtool = args.dimu, args.dimtool
        self.state_encoder = nn.Sequential(nn.Linear(self.dimtool, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 16))
        self.mlp = nn.Sequential(nn.Linear(1024+16, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(1024+16, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

        self.outputs = dict()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.apply(weight_init)

    def forward(self, data, detach_encoder=False):
        h_dough = self.encoder(data, detach=detach_encoder)
        h_tool = self.state_encoder(data['s_tool'])
        h = torch.cat([h_dough, h_tool], dim=-1)
        action = self.mlp(h)
        done = self.done_mlp(h)
        # hardcode for good initialization
        action = action / 5.
        done = done / 5.
        return action, done

class PointActor(nn.Module):
    """ PointCloud based actor"""

    def __init__(self, args, input_dim, action_dim):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(3)
        hidden_dim = args.actor_latent_dim
        self.dimu, self.dimtool = args.dimu, args.dimtool
        self.mlp = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

        self.outputs = dict()
        self.loss = torch.nn.MSELoss(reduction='none')
        self.apply(weight_init)

    def forward(self, data, detach_encoder=False):
        h = self.encoder(data, detach=detach_encoder)
        action = self.mlp(h)
        done = self.done_mlp(h)
        # hardcode for good initialization
        action = action / 5.
        done = done / 5.
        return action, done