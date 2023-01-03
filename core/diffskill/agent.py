import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from core.utils.simple_logger import LogDict
from core.utils import logger
from core.utils.diffskill_utils import batch_pred, dict_add_prefix, sample_n_ball, softclipping, weight_init
from core.models.pointnet_actor import PointActor, PointActorCat
from core.models.pointflow_vae import PointFlowVAE
from core.models.mlp_actor import ActorV0, ActorV1, ActorV2, ActorV3
from core.models.cnn_actor import CNNActor
from core.models.cnn_vae import CNNVAE
from core.models.mlp_abstraction import FeasibilityPredictor, RewardPredictor

class Agent(object):
    def __init__(self, args, solver, num_tools, device='cuda'):
        # """
        self.args = args
        self.solver = solver
        self.num_tools = num_tools
        self.dima, self.dimz = self.args.action_dim, self.args.dimz
        if args.input_mode == 'rgbd':
            self.dimu = self.dimz
            self.vae = CNNVAE(args, image_channels=len(args.img_mode), dimz=args.dimz, vae_beta=args.vae_beta).to(device)
            obs_shape = (args.img_size, args.img_size, 8)
            self.actors = nn.ModuleList([CNNActor(args, obs_shape,  action_dim=self.dima).to(device) for _ in range(num_tools)])
            self.feas = nn.ModuleList([FeasibilityPredictor(args, dimu=self.dimz).to(device) for _ in range(num_tools)])
            self.reward_predictor = RewardPredictor(args, dimu=self.dimz).to(device)
        elif args.input_mode == 'pc':
            self.dimu = self.dimz + 3  # Latent representation of the shape + translation
            self.dims = self.dimu + args.dimtool  # Dimension of the tool state
            self.args.dimu = self.dimu

            self.vae = PointFlowVAE(args, checkpoint_path=args.vae_resume_path)
            assert self.dimz == self.vae.pointflow_args.zdim

            # Policy takes s_t, u_g as input, where s = [z, t, z_tool], u_g = [z, t]
            actor_classes = {'v0': ActorV0, 'v1': ActorV1, 'v2': ActorV2, 'v3': ActorV3, 'pointnet_cat': PointActorCat, 'pointnet': PointActor}
            actor_class = actor_classes[args.actor_arch]
            self.actors = nn.ModuleList(
                [actor_class(args, input_dim=self.dims + self.dimu, action_dim=self.dima).to(device) for _ in range(num_tools)])
            self.feas = nn.ModuleList([FeasibilityPredictor(args, dimu=self.dimu).to(device) for _ in range(num_tools)])
            self.reward_predictor = RewardPredictor(args, dimu=self.dimu).to(device)

        if 'vae' in self.args.train_modules:
            self.optim = Adam(list(self.vae.parameters()), lr=args.rgb_vae_lr)
        else:
            self.optim = Adam(list(self.actors.parameters()) +
                              list(self.feas.parameters()) +
                              list(self.reward_predictor.parameters()),
                              lr=args.il_lr)
        self.terminate_early = True
        self.device = device

    def act(self, state, goal_dpc, tid):
        """ Batch inference! """
        n = state.shape[1]
        tool_state_idx = 3000 + tid * self.args.dimtool
        dpc = state[:, :3000].reshape(-1, 1000, 3)
        tool_state = state[:, tool_state_idx:tool_state_idx + self.args.dimtool]
        if n > 3300:
            num_primitives = 3 if 'Spread' in self.args.env_name or 'Gather' in self.args.env_name else 2
            tool_particle_idx = 3000 + num_primitives * self.args.dimtool + tid * 300
            tool_particles = state[:, tool_particle_idx:tool_particle_idx + 300].reshape(-1, 100, 3)
        z_obs, z_obs_trans = self.vae.encode(dpc)
        z_goal, z_goal_trans = self.vae.encode(goal_dpc)
        z_obs, z_obs_trans, z_goal, z_goal_trans = z_obs.detach(), z_obs_trans.detach(), z_goal.detach(), z_goal_trans.detach()
        u_obs, u_goal = torch.cat([z_obs, z_obs_trans], dim=1), torch.cat([z_goal, z_goal_trans], dim=1)
        if 'pointnet' not in self.args.actor_arch:
            obs = torch.cat([u_obs, tool_state, u_goal], dim=1)
            action, done = self.actors[tid](obs)
        elif self.args.actor_arch == 'pointnet_cat':
            action_data = {}
            obs = torch.cat([dpc, goal_dpc], dim=1)  # concat the channel for image, concat the batch for point cloud
            # Preprocess obs into the shape that encoder requires
            pos, feature = torch.split(obs, [3, obs.shape[-1] - 3], dim=-1)
            # dough: [0,1],target: [1,0]
            n = obs.shape[0]
            onehot = torch.zeros((obs.shape[0], obs.shape[1], 2)).to(obs.device)
            onehot[:, :1000, 1:2] += 1
            onehot[:, 1000:, 0:1] += 1
            x = torch.cat([feature, onehot], dim=-1)
            action_data['x'] = x.reshape(-1, 2 + feature.shape[-1])
            action_data['pos'] = pos.reshape(-1, 3)
            action_data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
            action_data['s_tool'] = tool_state
            action, done = self.actors[tid](action_data)
        else:
            obs = torch.cat([dpc, tool_particles, goal_dpc], dim=1)  # concat the batch for point cloud
            # Preprocess obs into the shape that encoder requires
            data = {}
            pos, feature = torch.split(obs, [3, obs.shape[-1] - 3], dim=-1)
            # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
            n = obs.shape[0]
            onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
            onehot[:, :1000, 2:3] += 1
            onehot[:, 1000:1100, 1:2] += 1
            onehot[:, 1100:, 0:1] += 1
            x = torch.cat([feature, onehot], dim=-1)
            data['x'] = x.reshape(-1, 3 + feature.shape[-1])
            data['pos'] = pos.reshape(-1, 3)
            data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
            action, done = self.actors[tid](data)

        return action, done, {'u_obs': u_obs, 'u_goal': u_goal}

    def act_rgbd(self, obs, goal, tid):
        """ Batch inference! """
        actions, dones = self.actors[tid](torch.cat([obs, goal], dim=1))
        return actions, dones, {}

    def act_ugoal(self, state, u_goal, tid):
        num_primitives = 3 if 'Spread' in self.args.env_name or 'Gather' in self.args.env_name else 2
        dpc = state[:, :3000].view(-1, 1000, 3)
        tool_state = state[:, 3000:3000 + num_primitives * self.args.dimtool].view(len(state), -1, self.args.dimtool)[:, tid, :]
        z_obs, z_obs_trans = self.vae.encode(dpc)
        z_obs, z_obs_trans = z_obs.detach(), z_obs_trans.detach()
        u_obs = torch.cat([z_obs, z_obs_trans], dim=1)
        obs = torch.cat([u_obs, tool_state, u_goal], dim=1)
        action, done = self.actors[tid](obs)
        return action, done, {'u_obs': u_obs, 'u_goal': u_goal}

    def random_negative_sample(self, N):
        cache = False if self.args.debug else True
        return self.vae.sample_u(N, cache=cache)

    def get_succ_loss(self, data_batch, tid, agent_id, noise):
        pos_idx_obs, pos_idx_goal, neg_idx_obs, neg_idx_goal = \
            data_batch['pos_idx_obs'], data_batch['pos_idx_goal'], data_batch['neg_idx_obs'], data_batch['neg_idx_goal']
        n_pos, n_neg, n_rand = len(pos_idx_obs[tid]), len(neg_idx_obs[tid]), self.args.num_random_neg
        u_pos_obs = self.vae.get_cached_encoding(pos_idx_obs[tid], noise=noise).to(self.device)
        u_pos_goal = self.vae.get_cached_encoding(pos_idx_goal[tid], noise=noise).to(self.device)
        u_neg_obs = self.vae.get_cached_encoding(neg_idx_obs[tid], noise=noise).to(self.device)
        u_neg_goal = self.vae.get_cached_encoding(neg_idx_goal[tid], noise=noise).to(self.device)
        u_rand_obs, u_rand_goal = torch.chunk(self.random_negative_sample(2 * n_rand).detach(), 2, dim=0)
        u_obs = torch.cat([u_pos_obs, u_neg_obs, u_rand_obs], dim=0)
        u_goal = torch.cat([u_pos_goal, u_neg_goal, u_rand_goal], dim=0)
        u_obs, u_goal = u_obs.detach(), u_goal.detach()

        # Weights between pos and neg is 1:1
        weight_pos, weight_neg = 0.5 / n_pos, 0.5 / (n_neg + n_rand)

        label = torch.zeros(n_pos + n_neg + n_rand, device=self.device, dtype=torch.float32)
        label[:n_pos] = 1.
        w_pos = torch.ones(n_pos, device=self.device, dtype=torch.float) * weight_pos
        w_neg = torch.ones(n_neg + n_rand, device=self.device, dtype=torch.float) * weight_neg
        weights = torch.cat([w_pos, w_neg], dim=0)  # Assume positive first
        pred_succ = batch_pred(self.feas[agent_id], {'obs': u_obs, 'goal': u_goal, 'eval': False})
        if self.args.bin_succ:
            label = label.to(torch.long)  # nn.CrossEntropyLoss requires
        succ_loss = self.feas[agent_id].loss(pred_succ, label)
        assert succ_loss.shape == weights.shape
        succ_loss = (succ_loss * weights).sum() / pred_succ.shape[0]
        with torch.no_grad():
            if self.args.bin_succ:
                pred_probs = self.feas[agent_id].softmax(pred_succ)
                avg_pred_pos, avg_pred_neg = torch.mean(pred_probs[:n_pos, 1]), torch.mean(pred_probs[n_pos:, 1])
                pred_labels = pred_probs[:, 1] > 0.5
            else:
                pred_probs = pred_succ
                avg_pred_pos, avg_pred_neg = torch.mean(pred_probs[:n_pos]), torch.mean(pred_probs[n_pos:])
                pred_labels = pred_probs > 0.5
            pred_pos_accuracy = torch.sum(pred_labels[:n_pos] == label[:n_pos]) / n_pos
            pred_neg_accuracy = torch.sum(pred_labels[n_pos:] == label[n_pos:]) / (n_neg + n_rand)

        d = {'succ_loss': succ_loss, 'avg_succ_pos': avg_pred_pos, 'avg_succ_neg': avg_pred_neg,
             'succ_pos_accuracy': pred_pos_accuracy,
             'succ_neg_accuracy': pred_neg_accuracy}
        return succ_loss, d

    def train_rgbd(self, data_batch, agent_ids=None, mode='train', epoch=None):
        log_dict = LogDict()
        tids = list(range(self.args.num_tools))
        if agent_ids is None:
            agent_ids = tids

        if 'vae' in self.args.train_modules:
            for tid, agent_id in zip(tids, agent_ids):
                obses, goals = data_batch['obses_rgb'], data_batch['goals_rgb']
                z_obs, mu_obs, logvar_obs = self.vae.encode(obses[tid])
                z_goal, mu_goal, logvar_goal = self.vae.encode(goals[tid])
                N = len(z_obs) + len(z_goal)
                sample_idx = np.random.choice(range(N), size=len(z_obs), replace=False)
                all_z = torch.cat([z_obs, z_goal], dim=0)[sample_idx]
                all_mu = torch.cat([mu_obs, mu_goal], dim=0)[sample_idx]
                all_logvar = torch.cat([logvar_obs, logvar_goal], dim=0)[sample_idx]
                all_original = torch.cat([obses[tid], goals[tid]], dim=0)[sample_idx]
                all_reconstr = self.vae.decode(all_z)
                vae_loss, vae_bce_loss, vae_kl_loss = self.vae.loss_fn(all_reconstr, all_original, all_mu, all_logvar)
                d = {'vae_loss': vae_loss,
                     'vae_bce_loss': vae_bce_loss,
                     'vae_kl_loss': vae_kl_loss, }
                log_dict.log_dict(d)
        else:
            noise = mode == 'train'
            obses_idx, goal_obses_idx = data_batch['obses_idx'], data_batch['goal_obses_idx']
            target_vs = data_batch['target_vs']
            dones, actions, score_labels, hindsight_flags = data_batch['dones'], data_batch['actions'], \
                                                            data_batch['score_labels'], data_batch['hindsight_flags']

            def get_hindsight_zgoal():
                z_hindsight_goal = self.vae.get_cached_encoding(goal_obses_idx[tid], noise=noise).to(self.device)  # Hindsight goal
                z_ori_goal = self.vae.get_cached_encoding(target_vs[tid], noise=noise, goal=True).to(self.device)  # Original goal
                flag = hindsight_flags[tid].view(-1, 1)
                z_goal = flag * z_hindsight_goal + (1 - flag) * z_ori_goal
                return z_goal

            for tid, agent_id in zip(tids, agent_ids):
                z_obs = self.vae.get_cached_encoding(obses_idx[tid], noise=noise).to(self.device)
                z_goal = get_hindsight_zgoal()
                z_obs, z_goal = z_obs.detach(), z_goal.detach()

                succ_loss = score_loss = action_loss = done_loss = 0.
                succ_info = {}

                if 'fea' in self.args.train_modules:
                    assert self.args.fea_type == 'regression'
                    # With a probability, replace negative samples with a random sample
                    succ_loss, succ_info = self.get_succ_loss(data_batch, tid, agent_id, noise)

                if 'reward' in self.args.train_modules:
                    pred_score = batch_pred(self.reward_predictor, {'obs': z_obs, 'goal': z_goal, 'eval': False})
                    score_loss = self.reward_predictor.loss(pred_score, score_labels[tid].flatten())  # Not reduced
                    score_loss = torch.sum(score_loss * (1. - hindsight_flags[tid])) / torch.sum(
                        (1. - hindsight_flags[tid]))  # No score loss for hindsight goals
                    if torch.sum((1. - hindsight_flags[tid])) == 0.:
                        score_loss = 0.

                if 'policy' in self.args.train_modules:
                    obses_rgb, goals_rgb = data_batch['obses_rgb'][tid], data_batch['goals_rgb'][tid]
                    obses_rgb = (torch.rand(obses_rgb.shape, device=obses_rgb.device) - 0.5) * 2 * self.args.obs_noise + obses_rgb
                    goals_rgb = (torch.rand(goals_rgb.shape, device=goals_rgb.device) - 0.5) * 2 * self.args.obs_noise + goals_rgb

                    pred_actions, pred_dones = self.actors[agent_id](torch.cat([obses_rgb, goals_rgb], dim=1))
                    action_loss = self.actors[0].loss(pred_actions, actions[tid]).sum() / pred_actions.shape[0]
                    done_loss = self.actors[0].loss(pred_dones, dones[tid][:, None]).sum() / pred_actions.shape[0]

                d = {'action_loss': action_loss,
                     'done_loss': done_loss,
                     'succ_loss': succ_loss,
                     'score_loss': score_loss}
                d.update(**succ_info)
                log_dict.log_dict(d)

        sum_dict = log_dict.agg(reduction='sum', numpy=False)
        if 'vae' in self.args.train_modules:
            l = sum_dict['vae_loss']
        else:
            l = self.args.weight_actor * sum_dict['action_loss'] + sum_dict['done_loss'] + \
                self.args.weight_fea * sum_dict['succ_loss'] + \
                self.args.weight_reward_predictor * sum_dict['score_loss']

        if mode == 'train':
            self.optim.zero_grad()
            l.backward()
            self.optim.step()
        ret_dict = log_dict.agg(reduction='mean', numpy=True)
        ret_dict = dict_add_prefix(ret_dict, 'avg_', skip_substr='sgld')
        return ret_dict

    def train_fea_pc(self, data_batch, agent_ids=None, mode='train', epoch=None):
        log_dict = LogDict()
        tids = list(range(self.args.num_tools))
        if agent_ids is None:  # TODO What is the difference between agent id and tid?
            agent_ids = tids

        assert self.args.fea_type == 'regression'
        assert not self.args.back_prop_encoder  # Freeze VAE weights
        noise = mode == 'train'
        for tid, agent_id in zip(tids, agent_ids):
            succ_loss, d = self.get_succ_loss(data_batch, tid, agent_id, noise=noise)
            log_dict.log_dict(d)
        sum_dict = log_dict.agg(reduction='sum', numpy=False)
        l = self.args.weight_fea * sum_dict['succ_loss']

        if mode == 'train':
            self.optim.zero_grad()
            l.backward()
            self.optim.step()
        ret_dict = log_dict.agg(reduction='mean', numpy=True)
        ret_dict = dict_add_prefix(ret_dict, 'avg_', skip_substr='sgld')
        return ret_dict

    def train_pc(self, data_batch, agent_ids=None, mode='train', epoch=None):
        obses_dpc, goal_obses_dpc, obses_tool = data_batch['obses_dpc'], data_batch['goal_obses_dpc'], data_batch['obses_tool']
        obses_idx, goal_obses_idx = data_batch['obses_idx'], data_batch['goal_obses_idx']
        target_vs = data_batch['target_vs']
        dones, actions, score_labels, hindsight_flags = data_batch['dones'], data_batch['actions'], \
                                                        data_batch['score_labels'], data_batch['hindsight_flags']
        log_dict = LogDict()
        tids = list(range(self.args.num_tools))
        if agent_ids is None:
            agent_ids = tids
        noise = mode == 'train'
        for tid, agent_id in zip(tids, agent_ids):
            assert tid == agent_id
            u_obs = self.vae.get_cached_encoding(obses_idx[tid], noise=noise).to(self.device)

            def get_hindsight_ugoal():
                u_hindsight_goal = self.vae.get_cached_encoding(goal_obses_idx[tid], noise=noise).to(self.device)  # Hindsight goal
                u_ori_goal = self.vae.get_cached_encoding(target_vs[tid], noise=noise, goal=True).to(self.device)  # Original goal
                flag = hindsight_flags[tid].view(-1, 1)
                u_goal = flag * u_hindsight_goal + (1 - flag) * u_ori_goal
                return u_goal

            u_goal = get_hindsight_ugoal()

            assert not self.args.back_prop_encoder  # Freeze VAE weights
            u_obs, u_goal = u_obs.detach(), u_goal.detach()

            if 'policy' in self.args.train_modules:
                if 'pointnet' not in self.args.actor_arch:
                    if noise:  # Noise for the actor
                        u_noise = (torch.rand((2, u_obs.shape[0], u_obs.shape[1]), device=u_obs.device) - 0.5) * 2 * self.args.actor_z_noise
                        u_noise[:, :, -3:] = (torch.rand(u_noise[:, :, -3:].shape, device=u_obs.device) - 0.5) * 2 * self.args.actor_t_noise
                        u_obs_actor = u_obs + u_noise[0]
                        u_goal_actor = u_goal + u_noise[1]
                    else:
                        u_obs_actor = u_obs
                        u_goal_actor = u_goal
                    s_tool = obses_tool[tid]
                    s_obs = torch.cat([u_obs_actor, s_tool], dim=1)

                    # Action loss
                    pred_actions, pred_dones = self.actors[agent_id](torch.cat([s_obs, u_goal_actor], dim=1))
                elif self.args.actor_arch == 'pointnet_cat':  # use pcl actor concat tool state
                    if noise:
                        obs_actor = (torch.rand(obses_dpc[tid].shape, device=obses_dpc[tid].device) - 0.5) * 2 * self.args.obs_noise + obses_dpc[tid]
                        goal_actor = (torch.rand(goal_obses_dpc[tid].shape, device=goal_obses_dpc[tid].device) - 0.5) * 2 * self.args.obs_noise + \
                                    goal_obses_dpc[tid]
                    else:
                        obs_actor = obses_dpc[tid]
                        goal_actor = goal_obses_dpc[tid]
                    # Action loss
                    action_data = {}
                    obs = torch.cat([obs_actor, goal_actor], dim=1)  # concat the channel for image, concat the batch for point cloud
                    # Preprocess obs into the shape that encoder requires
                    pos, feature = torch.split(obs, [3, obs.shape[-1] - 3], dim=-1)
                    # dough: [0,1],target: [1,0]
                    n = obs.shape[0]
                    onehot = torch.zeros((obs.shape[0], obs.shape[1], 2)).to(obs.device)
                    onehot[:, :1000, 1:2] += 1
                    onehot[:, 1000:, 0:1] += 1
                    x = torch.cat([feature, onehot], dim=-1)
                    action_data['x'] = x.reshape(-1, 2 + feature.shape[-1])
                    action_data['pos'] = pos.reshape(-1, 3)
                    action_data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
                    action_data['s_tool'] = obses_tool[tid]
                    pred_actions, pred_dones = self.actors[agent_id](action_data)
                else:  # pointcloud actor
                    obs_tool_particles = data_batch['obses_tool_particles']
                    if noise:
                        obs_actor = (torch.rand(obses_dpc[tid].shape, device=obses_dpc[tid].device) - 0.5) * 2 * self.args.obs_noise + obses_dpc[tid]
                        obs_tool_particles = (torch.rand(obs_tool_particles[tid].shape,
                                                        device=obs_tool_particles[tid].device) - 0.5) * 2 * self.args.obs_noise + obs_tool_particles[tid]
                        goal_actor = (torch.rand(goal_obses_dpc[tid].shape, device=goal_obses_dpc[tid].device) - 0.5) * 2 * self.args.obs_noise + \
                                    goal_obses_dpc[tid]
                    else:
                        obs_actor = obses_dpc[tid]
                        goal_actor = goal_obses_dpc[tid]
                        obs_tool_particles = obs_tool_particles[tid]
                    # Action loss
                    obs = torch.cat([obs_actor, obs_tool_particles, goal_actor], dim=1)  # concat the batch for point cloud
                    # Preprocess obs into the shape that encoder requires
                    data = {}
                    pos, feature = torch.split(obs, [3, obs.shape[-1] - 3], dim=-1)
                    # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
                    n = obs.shape[0]
                    onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
                    onehot[:, :1000, 2:3] += 1
                    onehot[:, 1000:1100, 1:2] += 1
                    onehot[:, 1100:, 0:1] += 1
                    x = torch.cat([feature, onehot], dim=-1)
                    data['x'] = x.reshape(-1, 3 + feature.shape[-1])
                    data['pos'] = pos.reshape(-1, 3)
                    data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
                    pred_actions, pred_dones = self.actors[agent_id](data)

                action_loss = self.actors[0].loss(pred_actions, actions[tid]).sum() / pred_actions.shape[0]
                done_loss = self.actors[0].loss(pred_dones, dones[tid][:, None]).sum() / pred_actions.shape[0]
            else:
                action_loss = 0.
                done_loss = 0.


            # Score loss
            pred_score = batch_pred(self.reward_predictor, {'obs': u_obs, 'goal': u_goal, 'eval': False})
            score_loss = self.reward_predictor.loss(pred_score, score_labels[tid].flatten())  # Not reduced
            score_loss = torch.sum(score_loss * (1. - hindsight_flags[tid])) / torch.sum(
                (1. - hindsight_flags[tid]))  # No score loss for hindsight goals
            if torch.sum((1. - hindsight_flags[tid])) == 0.:
                score_loss = 0.

            # Fea loss
            assert self.args.fea_type == 'regression'
            # With a probability, replace negative samples with a random sample
            succ_loss, succ_info = self.get_succ_loss(data_batch, tid, agent_id, noise)

            d = {'action_loss': action_loss,
                 'done_loss': done_loss,
                 'succ_loss': succ_loss,
                 'score_loss': score_loss}
            d.update(**succ_info)
            log_dict.log_dict(d)
        sum_dict = log_dict.agg(reduction='sum', numpy=False)
        l = self.args.weight_actor * sum_dict['action_loss'] + sum_dict['done_loss'] + \
            self.args.weight_fea * sum_dict['succ_loss'] + \
            self.args.weight_reward_predictor * sum_dict['score_loss']

        if mode == 'train':
            self.optim.zero_grad()
            l.backward()
            self.optim.step()
        ret_dict = log_dict.agg(reduction='mean', numpy=True)
        ret_dict = dict_add_prefix(ret_dict, 'avg_', skip_substr='sgld')
        return ret_dict

    def train(self, *args, **kwargs):
        if self.args.input_mode == 'rgbd':
            return self.train_rgbd(*args, **kwargs)
        else:
            return self.train_pc(*args, **kwargs)

    def save(self, path):
        torch.save({'actors': self.actors.state_dict(), 'feas': self.feas.state_dict(),
                    'reward': self.reward_predictor.state_dict(),
                    'vae': self.vae.state_dict()}, path)

    def load(self, path, modules=('policy', 'fea', 'reward', 'vae')):
        ckpt = torch.load(path)
        if 'policy' in modules:
            self.actors.load_state_dict(ckpt['actors'])
        if 'fea' in modules:
            self.feas.load_state_dict(ckpt['feas'])
        if 'reward' in modules:
            self.reward_predictor.load_state_dict(ckpt['reward'])
        if 'vae' in modules:
            if self.args.input_mode == 'pc':
                self.vae.model.load_state_dict(ckpt['vae']) # VAE separately loaded
            else:
                self.vae.load_state_dict(ckpt['vae'])
                # print(self.vae.state_dict())
        print('Agent {} loaded from {}'.format(modules, path))

    def load_vae(self, path):
        ckpt = torch.load(path)
        self.vae.load_state_dict(ckpt['vae'])
        print('Agent VAE loaded from ' + path)

    def load_actor(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        print('Actors loaded from ', path)

    def load_actor_reward(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        self.reward_predictor.load_state_dict(ckpt['reward'])
        print('Actors and reward loaded from ', path)

    def update_best_model(self, epoch, eval_info):
        if not hasattr(self, 'best_actor'):
            self.best_actor = BestModel(self.actors, os.path.join(logger.get_dir(), 'best_actor.ckpt'), 'actors')
            self.best_fea = BestModel(self.feas, os.path.join(logger.get_dir(), 'best_fea.ckpt'), 'feas')
            self.best_reward = BestModel(self.reward_predictor, os.path.join(logger.get_dir(), 'best_reward.ckpt'), 'reward')
        self.best_actor.update(epoch, eval_info.get('eval/avg_action_loss_mean', 0.))
        self.best_fea.update(epoch, eval_info.get('eval/avg_succ_loss_mean', 0.))
        self.best_reward.update(epoch, eval_info.get('eval/avg_score_loss_mean', 0.))
        self.best_dict = {'eval/best_actor_epoch': self.best_actor.best_epoch,
                          'eval/best_fea_epoch': self.best_fea.best_epoch,
                          'eval/best_reward_epoch': self.best_reward.best_epoch}

    def load_best_model(self):
        # Save training model
        self.training_model_param = {
            'actor': copy.deepcopy(self.actors.state_dict()),
            'fea': copy.deepcopy(self.feas.state_dict().copy()),
            'reward': copy.deepcopy(self.reward_predictor.state_dict())}
        # Use best model
        self.actors.load_state_dict(self.best_actor.param)
        self.feas.load_state_dict(self.best_fea.param)
        self.reward_predictor.load_state_dict(self.best_reward.param)

    def load_training_model(self):
        self.actors.load_state_dict(self.training_model_param['actor'])
        self.feas.load_state_dict(self.training_model_param['fea'])
        self.reward_predictor.load_state_dict(self.training_model_param['reward'])

class BestModel(object):
    def __init__(self, model, save_path, name):
        self.model = model
        self.save_path = save_path
        self.name = name

    def update(self, epoch, loss):
        if not hasattr(self, 'best_loss') or (loss < self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.param = copy.deepcopy(self.model.state_dict())
            torch.save({self.name: self.param}, self.save_path)
