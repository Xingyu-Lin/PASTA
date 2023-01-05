import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from core.models.pointnet_actor import PointActor, PointActorCat
from core.utils.diffskill_utils import batch_pred, dict_add_prefix
from core.utils.simple_logger import LogDict
from core.utils import logger
import os
import json
import copy
from core.models.pointflow_vae import PointFlowVAE
from core.models.mlp_actor import ActorV0, ActorV1, ActorV2, ActorV3, ActorV4
from core.utils.pasta_utils import match_set, match_set_pcl, env_skills, env_skills_unfilter
from core.models.set_cost_predictor import SetCostPredictor
from core.models.set_fea_predictor import SetFeasibilityPredictor
from core.pasta.plan.set_trajs import batch_random_choice
from core.utils.open3d_utils import visualize_pcl_policy_input
from core.utils.torch_chamfer import compute_chamfer_distance


class SetAgent(object):
    def __init__(self, args, solver, num_tools, device='cuda'):
        #
        self.args = args
        self.solver = solver
        self.num_tools = num_tools

        self.skill_def = env_skills[args.env_name] if self.args.filter_set else env_skills_unfilter[args.env_name]  # Set input dims for each skill
        self.dima, self.dimz = self.args.action_dim, self.args.dimz
        self.dimu = self.dimz + 3  # Latent representation of the shape + translation
        self.args.dimu = self.dimu

        # self.dims = self.dimu + args.dimtool  # Dimension of the tool state

        # Create VAE
        self.vae = PointFlowVAE(args, checkpoint_path=args.vae_resume_path)
        assert self.dimz == self.vae.pointflow_args.zdim

        # Create ppolicy and feasibility
        # Policy takes s_t, u_g as input, where s = [z, t, z_tool], u_g = [z, t]
        actor_classes = {'v0': ActorV0, 'v1': ActorV1, 'v2': ActorV2, 'v3': ActorV3, 'v4':ActorV4, 'pointnet_cat': PointActorCat, 'pointnet': PointActor}
        actor_class = actor_classes[args.actor_arch]
        feas, actors = [], []
        for skill in self.skill_def:
            N_in, N_out = skill['in'], skill['out']
            actor_input = self.dimu * N_in + args.dimtool
            fea_input = self.dimu * (N_in + N_out)
            actors.append(actor_class(args, input_dim=actor_input, action_dim=self.dima).to(device))
            feas.append(SetFeasibilityPredictor(args, input_dim=fea_input).to(device))

        self.actors = nn.ModuleList(actors)
        self.feas = nn.ModuleList(feas)
        self.reward_predictor = SetCostPredictor(args, input_dim=2 * self.dimu).to(device)

        self.optim = Adam(list(self.actors.parameters()) +
                          list(self.feas.parameters()) +
                          list(self.reward_predictor.parameters()),
                          lr=args.il_lr)
        self.terminate_early = True
        self.device = device

    def act(self, state, dpc, goal_dpc, tid, tool_particles=None):
        """ state: env state
            dpc: filtered dpc
            goal_dpc: filtered goal_dpc
        """
        n = state.shape[1]
        tool_state_idx = 3000 + tid * self.args.dimtool
        tool_state = state[:, tool_state_idx:tool_state_idx + self.args.dimtool]
        if n > 3300:
            num_primitives = 3 if 'Spread' in self.args.env_name or 'Gather' in self.args.env_name else 2
            tool_particle_idx = 3000 + num_primitives * self.args.dimtool + tid * 300
            if tool_particles is None:
                tool_particles = state[:, tool_particle_idx:tool_particle_idx + 300].reshape(-1, 100, 3)

        if 'pointnet' not in self.args.actor_arch:
            obs = torch.cat([dpc, goal_dpc], dim=0)
            img = visualize_pcl_policy_input(dpc[0].detach().cpu().numpy(),None,goal_dpc[0].detach().cpu().numpy())
            u_obs_goal = self.vae.encode_u(obs)
            action, done = self.actors[tid](u_obs_goal)
        elif self.args.actor_arch == 'pointnet_cat':
            action_data = {}
            obs = torch.cat([dpc, goal_dpc], dim=1)  # concat the channel for image, concat the batch for point cloud
            img = visualize_pcl_policy_input(dpc[0].detach().cpu().numpy(),None,goal_dpc[0].detach().cpu().numpy())
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
            # visualize_point_cloud([dpc[0].detach().cpu().numpy(), tool_particles[0].detach().cpu().numpy(),
            #                                  goal_dpc[0].detach().cpu().numpy()])
            # breakpoint()
            img = visualize_pcl_policy_input(dpc[0].detach().cpu().numpy(), tool_particles[0].detach().cpu().numpy(),
                                             goal_dpc[0].detach().cpu().numpy())
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

        return action, done, img

    def act_ugoal(self, state, u_goal, tid):
        dpc = state[:, :3000].view(-1, 1000, 3)
        tool_state = state[:, 3000:].view(len(state), -1, self.args.dimtool)[:, tid, :]
        z_obs, z_obs_trans = self.vae.encode(dpc)
        z_obs, z_obs_trans = z_obs.detach(), z_obs_trans.detach()
        u_obs = torch.cat([z_obs, z_obs_trans], dim=1)
        obs = torch.cat([u_obs, tool_state, u_goal], dim=1)
        action, done = self.actors[tid](obs)
        return action, done, {'u_obs': u_obs, 'u_goal': u_goal}

    def get_gt_reward(self, u_obs, u_goal):
        B, Ko, _ = u_obs.shape
        _, Kg, _ = u_goal.shape
        assert Ko == Kg
        obs_pc, goal_pc = self.vae.decode(u_obs.reshape(-1, self.dimu), 200), self.vae.decode(u_goal.reshape(-1, self.dimu), 200)
        dist = compute_chamfer_distance(obs_pc, goal_pc)
        dist = dist.view(B, Ko)
        return dist

    def random_negative_sample(self, N):
        cache = False if self.args.debug else True
        if not self.args.fea_uniform_z:
            return self.vae.sample_u(N, cache=cache)
        else:
            zs = self.vae.sample_unif_z(N)
            trans = self.vae.sample_t(N)[:, 0, :]
            return torch.cat([zs, trans], dim=-1)

    def train_rgbd(self, data_batch, agent_ids=None, mode='train', epoch=None):
        raise NotImplementedError

    def add_normal_noise(self, u_obs_goal, fea_z_noise=None, fea_t_noise=None):
        if fea_z_noise is None:
            fea_z_noise = self.args.fea_z_noise
        if fea_t_noise is None:
            fea_t_noise = self.args.fea_t_noise
        u = u_obs_goal.view(-1, self.dimu)
        noise = torch.normal(mean=0, std=fea_z_noise, size=u.shape, device=u.device)
        noise[:, -3:] = torch.normal(mean=0, std=fea_t_noise, size=u[:, -3:].shape, device=u.device)
        return u_obs_goal + noise.view(u_obs_goal.shape)

    def get_cached_set_input(self, idxes_obs, idxes_goal, n_in, n_out, noise=False, goal=True, verbose=False):
        # If goal is True, the goal index will be interpreted as the real goals. Otherwise, it will be interpreted as index of
        # observation treated as goal
        pcs_obs, pcs_goal, us_obs, us_goal = [], [], [], []
        goal_buffer = self.vae.cached_buffer_goal if goal else self.vae.cached_buffer_obs
        for idx_obs, idx_goal in zip(idxes_obs, idxes_goal):
            pcs_obs.append(self.vae.all_pcs[idx_obs])
            pcs_goal.append(self.vae.all_pcs[idx_goal])
            us_obs.append(self.vae.cached_buffer_obs[idx_obs])
            us_goal.append(goal_buffer[idx_goal])
        if self.args.filter_set:
            u_obs_goal, info = match_set(pcs_obs, pcs_goal, us_obs, us_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200, verbose=verbose)
        else:
            u_obs_goal, info = match_set(pcs_obs, pcs_goal, us_obs, us_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200, verbose=verbose, thr=0.)
        u_obs_goal = torch.FloatTensor(u_obs_goal).to(self.device)
        if noise:
            u_obs_goal = self.add_normal_noise(u_obs_goal)
        return u_obs_goal, info

    def get_policy_set_input(self, idxes_obs, idxes_goal, hindsight_flags, target_vs, n_in, n_out, verbose=False):
        if 'pointnet' in self.args.actor_arch:
            pcs_obs, pcs_goal = [], []
            for idx_obs, idx_goal, flag, target_v in zip(idxes_obs, idxes_goal, hindsight_flags, target_vs):
                pcs_obs.append(self.vae.all_pcs[idx_obs])
                hindsight_goal = self.vae.goal_pcs[target_v]  # Hindsight goal
                ori_goal = self.vae.all_pcs[idx_goal]  # Original goal
                goal = flag * hindsight_goal + (1 - flag) * ori_goal
                pcs_goal.append(goal)
            if not self.args.filter_set or n_out == 2:  # Cut or not filter set
                thr = 0.
            else:
                thr = 5e-4
            obs_pcl, goal_pcl, info = match_set_pcl(self.args.actor_batch_size, pcs_obs, pcs_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200,
                                                    verbose=verbose, thr=thr)
            obs_pcl = torch.FloatTensor(obs_pcl).to(self.device)
            goal_pcl = torch.FloatTensor(goal_pcl).to(self.device)
            return obs_pcl, goal_pcl, info
        else:
            pcs_obs, pcs_goal, us_obs, us_goal = [], [], [], []
            for idx_obs, idx_goal, flag, target_v in zip(idxes_obs, idxes_goal, hindsight_flags, target_vs):
                hindsight_goal, hindsight_goal_u = self.vae.goal_pcs[target_v], self.vae.cached_buffer_goal[target_v]
                ori_goal, ori_goal_u = self.vae.all_pcs[idx_goal], self.vae.cached_buffer_obs[idx_goal]
                goal = flag * hindsight_goal + (1 - flag) * ori_goal
                u_goal = flag * hindsight_goal_u + (1 - flag) * ori_goal_u
                pcs_obs.append(self.vae.all_pcs[idx_obs])
                pcs_goal.append(goal)
                us_obs.append(self.vae.cached_buffer_obs[idx_obs])
                us_goal.append(u_goal)
            if n_in == 1 and n_out == 2:  # Cut
                thr = 0.
            else:
                thr = 5e-4
            u_obs_goal, info = match_set(pcs_obs, pcs_goal, us_obs, us_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200, 
                                        verbose=verbose, thr=thr)
            u_obs_goal = torch.FloatTensor(u_obs_goal).to(self.device)
            u_obs_goal = self.add_normal_noise(u_obs_goal)
            return u_obs_goal, info

    def get_succ_loss(self, data_batch, tid, agent_id, noise, epoch):
        pos_idx_obs, pos_idx_goal, neg_idx_obs, neg_idx_goal = \
            data_batch['pos_idx_obs'], data_batch['pos_idx_goal'], data_batch['neg_idx_obs'], data_batch['neg_idx_goal']
        n_pos, n_neg, n_rand = len(pos_idx_obs[tid]), len(neg_idx_obs[tid]), self.args.num_random_neg

        n_in, n_out = self.skill_def[tid]['in'], self.skill_def[tid]['out']

        u_pos_obs_goal, info = self.get_cached_set_input(pos_idx_obs[tid], pos_idx_goal[tid], n_in, n_out, noise=noise, goal=False, verbose=False)
        n_pos = len(info['pc_idx'])  # Update pos idx
        u_rand_obs_goal = self.random_negative_sample((n_in + n_out) * n_rand).view(n_rand, -1)
        # Hard negative
        if self.args.hard_negative_type is None:
            u_neg_obs_goal = self.random_negative_sample((n_in + n_out) * n_rand).view(n_rand, -1)
        elif self.args.hard_negative_type == 'obs_goal':  # Switch out one entity to be random
            k = n_in + n_out
            u_neg_obs_goal = u_pos_obs_goal.clone().repeat([10, 1])
            n_neg = len(u_neg_obs_goal)
            u_rand = self.random_negative_sample(n_neg).view(n_neg, -1)
            change_idx, keep_idx = batch_random_choice(k, 1, n_neg)
            u_neg_obs_goal = u_neg_obs_goal.view(n_neg, k, self.dimu)
            u_neg_obs_goal[range(n_neg), change_idx[:, 0], :] = u_rand
            u_neg_obs_goal[np.arange(n_neg).repeat(k - 1).reshape(-1, k - 1), keep_idx, :] = self.add_normal_noise( \
                u_neg_obs_goal[np.arange(n_neg).repeat(k - 1).reshape(-1, k - 1), keep_idx, :], fea_t_noise=self.args.fea_t_noise_hard,
                fea_z_noise=self.args.fea_z_noise_hard)
            u_neg_obs_goal = u_neg_obs_goal.view(n_neg, k * self.dimu).clone()
        else:
            raise NotImplementedError

        u_obs_goal = torch.cat([u_pos_obs_goal, u_neg_obs_goal, u_rand_obs_goal], dim=0)
        u_obs_goal = u_obs_goal.detach()

        # Weights between pos and neg is 1:1
        weight_pos, weight_hard_neg, weight_rand_neg = 0.5 / n_pos, 0.25 / n_neg, 0.25 / n_rand

        label = torch.zeros(n_pos + n_neg + n_rand, device=self.device, dtype=torch.float32)
        label[:n_pos] = 1.

        w_pos = torch.ones(n_pos, device=self.device, dtype=torch.float) * weight_pos
        w_hard_neg = torch.ones(n_neg, device=self.device, dtype=torch.float) * weight_hard_neg
        w_rand_neg = torch.ones(n_rand, device=self.device, dtype=torch.float) * weight_rand_neg
        weights = torch.cat([w_pos, w_hard_neg, w_rand_neg], dim=0)  # Assume positive first
        pred_succ = batch_pred(self.feas[agent_id], {'obs_goal': u_obs_goal, 'eval': False})

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
             'succ_neg_accuracy': pred_neg_accuracy,
             'n_pos': n_pos}
        return succ_loss, d

    def get_reward_loss(self, data_batch, tid, epoch):
        pcs_obs, us_obs, pcs_goal, us_goal = [], [], [], []
        idxes_obs, idxes_goal = data_batch['pos_idx_obs'][tid], data_batch['target_vs'][tid]
        for idx_obs, idx_goal in zip(idxes_obs, idxes_goal):
            c1, c2 = np.random.randint(0, len(self.vae.all_pcs[idx_obs])), np.random.randint(0, len(self.vae.goal_pcs[idx_goal]))
            pcs_obs.append(self.vae.all_pcs[idx_obs][c1][None])
            pcs_goal.append(self.vae.goal_pcs[idx_goal][c2][None])
            us_obs.append(self.vae.cached_buffer_obs[idx_obs][c1][None])
            us_goal.append(self.vae.cached_buffer_goal[idx_goal][c2][None])
        pcs_obs, us_obs = np.vstack(pcs_obs), np.vstack(us_obs)
        pcs_goal, us_goal = np.vstack(pcs_goal), np.vstack(us_goal)

        if self.args.balanced_reward_training:
            Npos, Nneg = len(idxes_obs), len(idxes_obs)
            assert len(pcs_obs) == len(pcs_goal) == Npos
            # Negatives
            idx_pos_1, idx_pos_2 = np.arange(Npos), np.arange(Npos)
            idx_neg_1, idx_neg_2 = np.random.choice(len(pcs_obs), Nneg), np.random.choice(len(pcs_obs), Nneg)
            idx1, idx2 = np.concatenate([idx_pos_1, idx_neg_1]), np.concatenate([idx_pos_2, idx_neg_2])
        else:
            N = 1024  # Batch size for reward predictor
            idx1, idx2 = np.random.choice(len(pcs_obs), N), np.random.choice(len(pcs_goal), N)
        downsample_idx = np.random.choice(pcs_obs.shape[1], 200)
        pc1, pc2 = torch.FloatTensor(pcs_obs[idx1][:, downsample_idx]).to(self.device), torch.FloatTensor(pcs_goal[idx2][:, downsample_idx]).to(
            self.device)
        u_obs_goal = torch.FloatTensor(np.hstack([us_obs[idx1], us_goal[idx2]])).to(self.device)

        # Compute Chamfer distance as the label
        dist_label = compute_chamfer_distance(pc1, pc2)
        pred_score = batch_pred(self.reward_predictor, {'u_obs_goal': u_obs_goal, 'eval': False})
        assert dist_label.shape == pred_score.shape
        all_score_loss = self.reward_predictor.loss(pred_score, dist_label)  # Not reduced
        score_loss = torch.sum(all_score_loss) / len(all_score_loss)
        return score_loss

    def train_fea_pc(self, data_batch, agent_ids=None, mode='train', epoch=None):
        log_dict = LogDict()
        tids = list(range(self.args.num_tools))
        if agent_ids is None:  # TODO What is the difference between agent id and tid?
            agent_ids = tids

        assert self.args.fea_type == 'regression'
        assert not self.args.back_prop_encoder  # Freeze VAE weights
        noise = mode == 'train'
        for tid, agent_id in zip(tids, agent_ids):
            succ_loss, d = self.get_succ_loss(data_batch, tid, agent_id, noise=noise, epoch=epoch)
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

    def train_policy_pc(self, data_batch, tid, agent_id, noise, epoch):
        obses_idx, goal_obses_idx, obses_tool, obs_tool_particles = data_batch['obses_idx'][tid], data_batch['goal_obses_idx'][tid], \
                                                                    data_batch['obses_tool'][tid], data_batch['obses_tool_particles'][tid]
        target_vs = data_batch['target_vs'][tid]
        hindsight_flags = data_batch['hindsight_flags'][tid].view(-1, 1).cpu().numpy()
        dones, actions = data_batch['dones'][tid], data_batch['actions'][tid]

        assert tid == agent_id
        assert not self.args.back_prop_encoder  # Freeze VAE weights
        n_in, n_out = self.skill_def[tid]['in'], self.skill_def[tid]['out']
        if 'pointnet' in self.args.actor_arch:
            obses, goal_obses, info = self.get_policy_set_input(obses_idx, goal_obses_idx, hindsight_flags, target_vs, n_in, n_out, verbose=False)
        
        else:
            u_obs_goal, info = self.get_policy_set_input(obses_idx, goal_obses_idx, hindsight_flags, target_vs, n_in, n_out, verbose=False)
        
        obses_tool, dones, actions = obses_tool[info['pc_idx']], dones[info['pc_idx']], actions[info['pc_idx']]
        if obs_tool_particles is not None:
            obs_tool_particles = obs_tool_particles[info['pc_idx']]

        if 'pointnet' not in self.args.actor_arch:
            pred_actions, pred_dones = self.actors[agent_id](u_obs_goal)
        elif self.args.actor_arch == 'pointnet_cat':  # use pcl actor concat tool state
            if noise:
                obs_actor = (torch.rand(obses.shape, device=obses.device) - 0.5) * 2 * self.args.obs_noise + obses
                goal_actor = (torch.rand(goal_obses.shape, device=goal_obses.device) - 0.5) * 2 * self.args.obs_noise + \
                             goal_obses
            else:
                obs_actor = obses
                goal_actor = goal_obses
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
            action_data['s_tool'] = obses_tool
            pred_actions, pred_dones = self.actors[agent_id](action_data)
        else:  # pointcloud actor
            if noise:
                obs_actor = (torch.rand(obses.shape, device=obses.device) - 0.5) * 2 * self.args.obs_noise + obses
                obs_tool_particles = (torch.rand(obs_tool_particles.shape,
                                                 device=obs_tool_particles.device) - 0.5) * 2 * self.args.obs_noise + obs_tool_particles
                goal_actor = (torch.rand(goal_obses.shape, device=goal_obses.device) - 0.5) * 2 * self.args.obs_noise + \
                             goal_obses
            else:
                obs_actor = obses
                goal_actor = goal_obses
                obs_tool_particles = obs_tool_particles
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

        action_loss = self.actors[0].loss(pred_actions, actions).sum() / pred_actions.shape[0]
        done_loss = self.actors[0].loss(pred_dones, dones[:, None]).sum() / pred_actions.shape[0]
        return action_loss, done_loss

    def train_pc(self, data_batch, agent_ids=None, mode='train', epoch=None):
        log_dict = LogDict()
        tids = list(range(self.args.num_tools))
        if agent_ids is None:  # TODO What is the difference between agent id and tid?
            agent_ids = tids
        noise = mode == 'train'
        for tid, agent_id in zip(tids, agent_ids):
            assert tid == agent_id
            if 'policy' in self.args.train_modules:
                action_loss, done_loss = self.train_policy_pc(data_batch, tid, agent_id, noise, epoch)
            else:
                action_loss = 0.
                done_loss = 0.

            if 'fea' in self.args.train_modules:
                assert self.args.fea_type == 'regression'
                succ_loss, succ_info = self.get_succ_loss(data_batch, tid, agent_id, noise, epoch)
            else:
                succ_loss = 0.
                succ_info = {}

            if 'reward' in self.args.train_modules:
                score_loss = self.get_reward_loss(data_batch, tid, epoch)
            else:
                score_loss = 0.

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

    def load(self, path, modules):
        ckpt = torch.load(path)
        if 'policy' in modules:
            self.actors.load_state_dict(ckpt['actors'])
        if 'fea' in modules:
            self.feas.load_state_dict(ckpt['feas'])
        if 'reward' in modules:
            self.reward_predictor.load_state_dict(ckpt['reward'])
        # self.vae.load_state_dict(ckpt['vae']) # VAE separately loaded
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
