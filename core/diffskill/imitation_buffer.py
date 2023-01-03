import torch
import numpy as np
import os
from core.utils.diffskill_utils import batch_rand_int, img_to_tensor, img_to_np
from core.diffskill.buffer import ReplayBuffer


def filter_buffer_nan(buffer):
    actions = buffer.buffer['actions']
    idx = np.where(np.isnan(actions))
    print('{} nan actions detected. making them zero.'.format(len(idx[0])))
    buffer.buffer['actions'][idx] = 0.


class ImitationReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(ImitationReplayBuffer, self).__init__(*args, **kwargs)
        self.tool_idxes = {}

    # Should also be consistent with the one when generating the feasibility prediction dataset! line 291 of train.py
    def get_tid(self, action_mask):
        """ Should be consistent with get_tool_idxes"""
        if len(action_mask.shape) == 1:
            return int(action_mask[0] < 0.5)
        elif len(action_mask.shape) == 2:
            return np.array(action_mask[:, 0] < 0.5).astype(np.int)

    def get_optimal_step(self):
        """ For each trajectory, compute its optimal time step - return [0, horizon] for each one """
        if not hasattr(self, 'optimal_step'):
            emd_rewards = -self.buffer['info_emds'][:self.cur_size].reshape(-1, self.horizon)
            N = emd_rewards.shape[0]
            # For each step in the buffer, find its optimal future step.
            optimal_step = np.zeros((N, self.horizon), dtype=np.int32)
            for i in range(N):
                max_iou = -1.
                max_step = None
                for j in reversed(range(self.horizon)):
                    if emd_rewards[i, j] >= max_iou or max_step is None:
                        max_step = j
                        max_iou = emd_rewards[i, j]
                    optimal_step[i, j] = max_step
            self.optimal_step = optimal_step
        return self.optimal_step

    def get_epoch_tool_idx(self, epoch, tid, mode):
        # Get a index generator for each tool index of the mini-batch
        # Note: Assume static buffer
        assert mode in ['train', 'eval']
        cache_name = f'{mode}_{tid}'
        if cache_name not in self.tool_idxes:
            idxes = self.train_idx if mode == 'train' else self.eval_idx
            tool_idxes = self.get_tool_idxes(tid, idxes)
            # Shuffle the tool idxes so that if the trajectories in the replay buffer is ordered, this will shuffle the order.
            tool_idxes = tool_idxes.reshape(-1, 50)
            perm = np.random.permutation(len(tool_idxes))
            tool_idxes = tool_idxes[perm].flatten()
            self.tool_idxes[cache_name] = tool_idxes
        if mode == 'train':
            num_steps = min((self.args.step_per_epoch * epoch + self.args.step_warmup) // 2,
                            len(self.tool_idxes[cache_name]))  # Divide by 2 because there are 2 tools
            B = self.args.batch_size
        else:
            num_steps = len(self.tool_idxes[cache_name])
            B = 512 if 'pointnet' not in self.args.actor_arch else self.args.batch_size  # Larger batch size for faster evaluation, pointcloud input cannot use a large batchsize
        permuted_idx = self.tool_idxes[cache_name][np.random.permutation(num_steps)]
        epoch_tool_idxes = [permuted_idx[np.arange(i, min(i + B, num_steps))] for i in range(0, num_steps, B)]
        return epoch_tool_idxes

    def sample_goal(self, obs_idx, hindsight_goal_ratio, device):
        n = len(obs_idx)
        horizon = 50
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        init_v, target_v = self.buffer['init_v'][obs_idx], self.buffer['target_v'][obs_idx]
        hindsight_future_idx = batch_rand_int(traj_t, horizon, n)
        optimal_step = self.get_optimal_step()
        hindsight_flag = np.round(np.random.random(n) < hindsight_goal_ratio).astype(np.int32)
        if self.args.input_mode == 'rgbd':
            hindsight_goal_imgs = self.buffer['obses'][hindsight_future_idx + traj_id * horizon]
            target_goal_imgs = self.np_target_imgs[target_v]
            goal_obs = img_to_tensor(hindsight_flag[:, None, None, None] * hindsight_goal_imgs +
                                     (1 - hindsight_flag[:, None, None, None]) * target_goal_imgs, mode=self.args.img_mode).to(device,
                                                                                                                               non_blocking=True)
        elif self.args.input_mode == 'pc':
            hindsight_goal_states = self.buffer['states'][hindsight_future_idx + traj_id * horizon]
            hindsight_goal_states = hindsight_goal_states[:, :3000].reshape(n, -1, 3)
            target_goal_grid = self.np_target_pc[target_v]
            goal_obs = torch.FloatTensor(hindsight_flag[:, None, None] * hindsight_goal_states +  # replace target goal imgs by target point cloud
                                         (1 - hindsight_flag[:, None, None]) * target_goal_grid).to(device, non_blocking=True)
        else:
            raise NotImplementedError
        hindsight_done_flag = (hindsight_future_idx == 0).astype(np.int32)
        target_done_flag = (traj_t == optimal_step[traj_id, traj_t]).astype(np.int32)
        done_flag = hindsight_flag * hindsight_done_flag + (1 - hindsight_flag) * target_done_flag
        done_flag = torch.FloatTensor(done_flag).to(device, non_blocking=True)
        hindsight_flag = torch.FloatTensor(hindsight_flag).to(device, non_blocking=True)
        return goal_obs, done_flag, hindsight_flag

    def sample_goal_idx(self, obs_idx, hindsight_goal_ratio, device):
        n = len(obs_idx)
        horizon = 50

        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        target_v = self.buffer['target_v'][obs_idx]  # Used for getting u for the goal pointcloud
        hindsight_future_idx = batch_rand_int(traj_t, horizon, n)
        optimal_step = self.get_optimal_step()
        hindsight_flag = np.round(np.random.random(n) < hindsight_goal_ratio).astype(np.int32)
        goal_idx = hindsight_future_idx + traj_id * horizon

        hindsight_done_flag = (hindsight_future_idx == 0).astype(np.int32)
        target_done_flag = (traj_t == optimal_step[traj_id, traj_t]).astype(np.int32)
        done_flag = hindsight_flag * hindsight_done_flag + (1 - hindsight_flag) * target_done_flag
        done_flag = torch.FloatTensor(done_flag).to(device, non_blocking=True)
        hindsight_flag_tensor = torch.FloatTensor(hindsight_flag).to(device, non_blocking=True)
        return goal_idx, done_flag, (hindsight_flag, hindsight_flag_tensor), target_v

    def sample_positive_idx(self, obs_idx):
        n = len(obs_idx)
        horizon = 50
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        future_idx = batch_rand_int(traj_t, horizon, n) + traj_id * horizon
        return future_idx

    def sample_negative_idx(self, obs_idx, epoch):
        horizon = 50
        # Number of trajectories used in the first few epochs, which can be used for negative sampling
        num_traj = min(self.args.step_per_epoch * (epoch + 1) + self.args.step_warmup, len(self)) // horizon
        assert num_traj > 1
        n = len(obs_idx)
        traj_id = obs_idx // horizon
        neg_traj_id = (traj_id + np.random.randint(1, num_traj)) % num_traj
        traj_t = np.random.randint(0, horizon, n)
        neg_idx = neg_traj_id * horizon + traj_t
        return neg_idx

    def sample_reset_obs(self, obs_idx):
        horizon = 50
        traj_id = obs_idx // horizon
        reset_lens = self.buffer['reset_motion_lens'][traj_id]

        did_reset_idx = np.where(reset_lens > 0)
        traj_id = traj_id[did_reset_idx]
        reset_lens = reset_lens[did_reset_idx]
        reset_idx = batch_rand_int(0, reset_lens, len(reset_lens))
        reset_imgs = self.buffer['reset_motion_obses'][traj_id, reset_idx]
        reset_states = self.buffer['reset_state'][traj_id, reset_idx]
        return did_reset_idx, reset_imgs, reset_states

    def compute_stats(self):
        pass

    def save_buffer_z(self, agent, save_path=None):
        from core.utils.diffskill_utils import batch_pred, img_to_tensor
        def get_zt(idx):
            pcs, _ = self.get_state(idx)
            with torch.no_grad():
                x = torch.FloatTensor(pcs).cuda()
                u = batch_pred(agent.vae.encode_u, {'x': x}, batch_size=128)
                z = u[:, :-3].detach().cpu()
                t = u[:, -3:].detach().cpu()
            return z, t

        (z_train, t_train), (z_eval, t_eval) = get_zt(self.train_idx), get_zt(self.eval_idx)
        d = {'z_train': z_train.numpy(), 'z_eval': z_eval.numpy(),
             't_train': t_train.numpy(), 't_eval': t_eval.numpy()}
        import pickle
        if save_path is None:
            save_path = self.load_path
        with open(os.path.join(save_path, 'buffer_stat.pkl'), 'wb') as f:
            pickle.dump(d, f)

    def get_state(self, idxes):
        num_primitives = 3 if 'Spread' in self.args.env_name or 'Gather' in self.args.env_name else 2
        if self.args.input_mode == 'pc':
            dpc = self.buffer['states'][idxes, :3000].reshape(-1, 1000, 3)
        else:
            dpc = self.buffer['dough_pcl'][idxes].reshape(-1, 1000, 3)
        tool_state = self.buffer['states'][idxes, 3000:3000+num_primitives*self.args.dimtool].reshape(len(idxes), -1)

        return dpc, tool_state

    def get_tool_particles(self, idxes, tid):
        n = self.buffer['states'].shape[1]
        assert n > 3300
        num_primitives = 3 if 'Spread' in self.args.env_name or 'Gather' in self.args.env_name else 2
        tool_particle_idx = 3000 + num_primitives*self.args.dimtool + tid*300
        tool_particles = self.buffer['states'][idxes, tool_particle_idx:tool_particle_idx+300].reshape(-1, 100, 3)
        return tool_particles

    def sample_tool_transitions_fea(self, batch_tool_idxes, epoch):
        ret = {}
        for key in ['pos_idx_obs', 'pos_idx_goal', 'neg_idx_obs', 'neg_idx_goal']:
            ret[key] = []

        for tid, curr_tool_idx in enumerate(batch_tool_idxes):
            ret['pos_idx_obs'].append(curr_tool_idx)
            ret['pos_idx_goal'].append(self.sample_positive_idx(curr_tool_idx))  # Batch size of positive example
            K = int(np.ceil(self.args.num_buffer_neg / len(curr_tool_idx)))
            neg_idx_obs = np.tile(curr_tool_idx[None, :], [K, 1]).flatten()[:self.args.num_buffer_neg]
            neg_idx_goal = self.sample_negative_idx(neg_idx_obs, epoch=epoch)
            ret['neg_idx_obs'].append(neg_idx_obs)
            ret['neg_idx_goal'].append(neg_idx_goal)
        return ret

    def sample_tool_transitions(self, batch_tool_idxes, epoch, device):
        """
        :param batch_tool_idxes: A list of index for one mini-batch for each tool
        :return: Dictionary of data in torch tensor, with each item being a list of the corresponding data for each tool
        """
        ret = {}
        for key in ['obses_idx',
                    'goal_obses_idx',
                    'obses_dpc',  # Current dough point cloud
                    'obses_tool',  # Current tool state
                    'obses_tool_particles',  # Current tool particles
                    'goal_obses_dpc',  # Goal dough point cloud
                    'obses_rgb',  # Current rgbd observation
                    'goals_rgb',  # Current target images
                    'actions',
                    'dones',
                    'score_labels',
                    'hindsight_flags',
                    'target_vs']:
            ret[key] = []

        for tid, curr_tool_idx in enumerate(batch_tool_idxes):
            # For BC
            assert self.args.frame_stack == 1
            if self.args.input_mode == 'rgbd':
                if 'vae' in self.args.train_modules:
                    obses = img_to_tensor(self.buffer['obses'][curr_tool_idx], mode=self.args.img_mode).to(device, non_blocking=True)
                    target_v = self.buffer['target_v'][curr_tool_idx]  # Used for getting u for the goal pointcloud
                    goals = self.target_imgs[target_v]
                    done, hindsight_flag = None, None
                    ret['obses_rgb'].append(obses)
                    ret['goals_rgb'].append(goals)
                else:
                    goal_obses_idx, done, (hindsight_flag_np, hindsight_flag), target_v = \
                        self.sample_goal_idx(curr_tool_idx, self.args.hindsight_goal_ratio, device)
                    ret['obses_idx'].append(curr_tool_idx)
                    ret['goal_obses_idx'].append(goal_obses_idx)
                    ret['target_vs'].append(target_v)

                    if 'policy' in self.args.train_modules:
                        obses = img_to_tensor(self.buffer['obses'][curr_tool_idx], mode=self.args.img_mode).to(device, non_blocking=True)
                        hindsight_goal_imgs = self.buffer['obses'][goal_obses_idx]
                        target_goal_imgs = self.np_target_imgs[target_v]
                        goal_obs = img_to_tensor(hindsight_flag_np[:, None, None, None] * hindsight_goal_imgs +
                                                 (1 - hindsight_flag_np[:, None, None, None]) * target_goal_imgs,
                                                 mode=self.args.img_mode).to(device, non_blocking=True)
                        ret['obses_rgb'].append(obses)
                        ret['goals_rgb'].append(goal_obs)
            elif self.args.input_mode == 'pc':
                obs_dpc, obs_tool = self.get_state(curr_tool_idx)
                goal_obses_idx, done, (hindsight_flag_np, hindsight_flag), target_v = \
                    self.sample_goal_idx(curr_tool_idx, self.args.hindsight_goal_ratio, device)
                if 'policy' in self.args.train_modules and not self.args.train_set:  # For trainset, set pc is already cached.
                    obs_dpc = torch.FloatTensor(obs_dpc).to(device, non_blocking=True)
                    hindsight_goal_states = self.buffer['states'][goal_obses_idx]
                    hindsight_goal_states = hindsight_goal_states[:, :3000].reshape(len(hindsight_goal_states), 1000, 3)
                    target_goal_grid = self.np_target_pc[target_v]
                    goal_dpc = torch.FloatTensor(
                        hindsight_flag_np[:, None, None] * hindsight_goal_states +  # replace target goal imgs by target point cloud
                        (1 - hindsight_flag_np[:, None, None]) * target_goal_grid).to(device, non_blocking=True)
                    # goal_dpc, done, hindsight_flag = self.sample_goal(curr_tool_idx, self.args.hindsight_goal_ratio, device)

                obs_tool = obs_tool.reshape(len(obs_tool), -1, self.args.dimtool)[:, tid, :]
                obs_tool = torch.FloatTensor(obs_tool).to(device, non_blocking=True)
                obs_tool_particles = None if self.args.actor_arch != 'pointnet' else self.get_tool_particles(curr_tool_idx, tid)
                if obs_tool_particles is not None:
                    obs_tool_particles = torch.FloatTensor(obs_tool_particles).to(device, non_blocking=True)

                if 'policy' in self.args.train_modules and not self.args.train_set:
                    ret['obses_dpc'].append(obs_dpc)
                    ret['goal_obses_dpc'].append(goal_dpc)
                ret['target_vs'].append(target_v)
                ret['obses_idx'].append(curr_tool_idx)
                ret['goal_obses_idx'].append(goal_obses_idx)

                ret['obses_tool'].append(obs_tool)
                ret['obses_tool_particles'].append(obs_tool_particles)

            else:
                raise NotImplementedError

            action = torch.FloatTensor(self.buffer['actions'][curr_tool_idx]).to(device, non_blocking=True)
            score_label = -torch.FloatTensor(self.buffer['info_emds'][curr_tool_idx]).to(device,
                                                                                         non_blocking=True)  # Negative of the EMD as the score
            ret['actions'].append(action)
            ret['dones'].append(done)
            ret['score_labels'].append(score_label)
            ret['hindsight_flags'].append(hindsight_flag)  # Scores labels come form f(o_curr, o_g) and do not apply to hindsight goals

        # Sample feasibility transtion
        succ_data = self.sample_tool_transitions_fea(batch_tool_idxes, epoch)
        ret.update(**succ_data)
        return ret
