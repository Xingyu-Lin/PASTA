import torch
import torch.nn as nn
from core.utils.diffskill_utils import weight_init
from core.models.cnn_encoder import CNNEncoder

class CNNActor(nn.Module):
    """ Image based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        self.args = args
        self.encoder = CNNEncoder(obs_shape, args.actor_feature_dim)
        latent_dim = args.actor_feature_dim
        self.action_mlp = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        return self.action_mlp(obs), self.done_mlp(obs)
