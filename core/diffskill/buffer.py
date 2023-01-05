import numpy as np
import pickle
import torch
import gzip
import os
from glob import glob
from core.utils.logger import reset
from core.utils.diffskill_utils import batch_rand_int, load_target_imgs, load_target_pcs


class ReplayBuffer(object):
    def __init__(self, args, maxlen=int(150000), her_args=None):
        self.args = args
        self.maxlen = maxlen

        self.cur_size = 0
        self.cur_pt = 0
        self.buffer = {}  # a dictionary of keys, each is an array of size N x dim
        # self.init_vs, self.target_vs, self.action_masks = [], [], []
        self.horizon = 50  # TODO Remove hard coding
        self.maxtraj = maxlen // self.horizon
        self.her_args = her_args
        if her_args is not None:
            self.reward_fn = her_args.reward_fn

    def get_tool_idxes(self, tid, idxes=None):
        """ Return categorized time index within the given idxes based on its tool"""
        action_mask = self.buffer['action_mask']
        if idxes is None:
            idxes = np.arange(self.cur_size)
        if self.args.env_name == 'CutRearrangeSpread-v1':
            if tid == 0:
                return idxes[np.where(action_mask[idxes, 0] > 0.5)[0]]
            elif tid == 2:
                return idxes[np.where(action_mask[idxes, -1] > 0.5)[0]]
            elif tid == 1:
                return idxes[np.where(np.logical_and(action_mask[idxes, -1] < 0.5, action_mask[idxes, 0] < 0.5))[0]]
        else:
            if tid == 0:
                return idxes[np.where(action_mask[idxes, 0] > 0.5)[0]]
            elif tid == 1:
                return idxes[np.where(action_mask[idxes, 0] < 0.5)[0]]

    def generate_train_eval_split(self, train_ratio=0.9, filter=False, cached_state_path=None):
        np.random.seed(42)
        if filter:
            n = np.arange(0, self.cur_size, self.horizon)
            init_vs, target_vs = self.buffer['init_v'][n], self.buffer['target_v'][n]
            tool_0, tool_1 = self.get_tool_idxes(0, idxes=n), self.get_tool_idxes(1, idxes=n) # first find the tool trajectories in the buffer
            if ('0202_gathermove' in cached_state_path) or (self.args is not None and '0202_gathermove' in self.args.cached_state_path):
                idx0 = n[np.where(np.logical_and(init_vs % 2 == 0, target_vs < 100))[0]] # find the correct tool trajectories based on init and target
                idx1 = n[np.where(np.logical_and(init_vs % 2 != 0, target_vs >= 100))[0]]
            elif ('1215_cutrearrange' in cached_state_path) or (self.args is not None and '1215_cutrearrange' in self.args.cached_state_path):
                idx0 = n[np.where(np.logical_and(init_vs % 3 == 0, target_vs % 3 == 0))[0]]
                idx1 = n[np.where(np.logical_and(init_vs % 3 != 0, init_vs % 3 == target_vs % 3))[0]]
            elif ('0202_liftspread' in cached_state_path) or (self.args is not None and '0202_liftspread' in self.args.cached_state_path):
                idx0 = n[np.where(np.logical_and(init_vs < 100, target_vs >= 100))[0]] # find the correct tool trajectories based on init and target
                idx1 = n[np.where(np.logical_and(init_vs >= 100, target_vs < 100))[0]]
            else:
                raise NotImplementedError
            filtered_0 = tool_0[np.where(np.isin(tool_0, idx0))[0]] // self.horizon # filter out tool idxes not in the correct init target combination
            filtered_1 = tool_1[np.where(np.isin(tool_1, idx1))[0]] // self.horizon
            traj_idxes = np.random.permutation(np.concatenate([filtered_0, filtered_1]))
            num_traj = len(traj_idxes)
        else:
            num_traj = len(self) // self.horizon
            traj_idxes = np.random.permutation(num_traj)
        num_train_traj = int(num_traj * train_ratio)
        assert not hasattr(self, 'train_traj_idx')
        self.train_traj_idx, self.eval_traj_idx = traj_idxes[:num_train_traj], traj_idxes[num_train_traj:]
        self.train_idx = (self.train_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        self.eval_idx = (self.eval_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()

        print(f"Number of trajectories: {num_traj}")
        print(f"Number of trajectories for train: {len(self.train_traj_idx)}")
        print(f"Number of trajectories for eval: {len(self.eval_traj_idx)}")

    def add(self, traj, save_state=True):
        if len(self.buffer) == 0:
            # initialize buffer
            for key in traj:
                if key == 'states':
                    if save_state:
                        self.buffer[key] = np.empty(shape=(self.maxlen, *traj[key].shape[1:]), dtype=np.float)
                    continue
                if 'reset' in key:
                    self.buffer[key] = np.empty(shape=(self.maxlen // self.horizon, *traj[key].shape[1:]), dtype=traj[key].dtype)
                    continue
                if key in ['init_v', 'target_v']:
                    self.buffer[key] = np.empty(shape=(self.maxlen), dtype=np.int32)
                    continue
                if key == 'action_mask' and len(np.array(traj[key]).shape) != 2:
                    self.buffer[key] = np.empty(shape=(self.maxlen, np.array(traj[key]).reshape(-1, 1).shape[0]), dtype=np.int32)
                    continue
                if key == 'mass_grid':
                    # Only save the first few for mass_grid due to memory limit
                    self.buffer[key] = np.empty(shape=(15000, *traj[key].shape[1:]), dtype=traj[key].dtype)
                    continue
                if not isinstance(traj[key], np.ndarray) or np.prod(traj[key].shape) == 1:  # Do not save scaler value in the buffer
                    continue
                # print(key, traj[key].shape)
                if key == 'ious':
                    T = traj['ious'].shape[0]
                    self.buffer['ious'] = np.empty(shape=(self.maxlen, T), dtype=traj[key].dtype)
                else:
                    # print('key:', key, traj[key].shape[1:])
                    self.buffer[key] = np.empty(shape=(self.maxlen, *traj[key].shape[1:]), dtype=traj[key].dtype)
        N = traj['actions'].shape[0]

        if self.cur_size + N < self.maxlen:
            idxes = np.arange(self.cur_size, self.cur_size + N)
            reset_idx = np.arange(self.cur_size // self.horizon, (self.cur_size + N) // self.horizon)
            self.cur_size += N
            self.cur_pt = self.cur_size
        else:  # full
            idxes = np.arange(self.cur_pt, self.cur_pt + N) % self.maxlen
            if 'reset_motion_obses' in self.buffer:
                reset_idx = np.arange(self.cur_size // self.horizon, (self.cur_size + N) // self.horizon) % self.buffer['reset_motion_obses'].shape[0]
            self.cur_size = self.maxlen
            self.cur_pt = (self.cur_pt + N) % self.maxlen

        if 'reset_motion_obses' in self.buffer:
            self.buffer['reset_states'][reset_idx] = traj['reset_states']
            self.buffer['reset_motion_obses'][reset_idx] = traj['reset_motion_obses']
            self.buffer['reset_motion_lens'][reset_idx] = traj['reset_motion_lens']
            self.buffer['reset_info_emds'][reset_idx] = traj['reset_info_emds']

        if 'states' in self.buffer:
            self.buffer['states'][idxes] = traj['states'][:-1]
        if 'obses' in self.buffer:
            self.buffer['obses'][idxes] = traj['obses'][:-1]
        self.buffer['actions'][idxes] = traj['actions']
        self.buffer['rewards'][idxes] = traj['rewards']
        # self.buffer['mass_grid'][idxes] = traj['mass_grid'][:-1]
        # self.buffer['ious'][idxes] = traj['ious']  # Already remove the last one when computing the pairwise iou
        # self.buffer['target_ious'][idxes] = traj['target_ious'][:-1]
        if 'info_emds' in self.buffer:
            self.buffer['info_emds'][idxes] = traj['info_emds'][:-1]
        if 'info_normalized_performance' in self.buffer:
            self.buffer['info_normalized_performance'][idxes] = traj['info_normalized_performance'][:-1]
        self.buffer['init_v'][idxes] = traj['init_v']
        self.buffer['target_v'][idxes] = traj['target_v']
        self.buffer['action_mask'][idxes] = traj['action_mask'][None]

    def sample(self, B):
        idx = np.random.randint(0, self.cur_size, B)
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][idx]
        return batch

    def get_goal(self, target_v):
        """ Get goal obs from target_v"""
        if not hasattr(self, 'np_target_imgs'):
            self.np_target_imgs = load_target_imgs(self.her_args.cached_state_path, ret_tensor=False)
            self.np_target_pc = load_target_pcs(self.her_args.cached_state_path)

        return self.np_target_imgs[target_v], self.np_target_pc[target_v]

    def get_traj(self, n):
        # get a specific trajectory in the buffer
        if n >= self.cur_size // self.horizon:
            assert False, "Out of buffer's max index"
        else:
            start_idx = n * self.horizon
            end_idx = start_idx + self.horizon
            traj = {}
            for key in self.buffer:
                if 'reset' in key:
                    traj[key] = self.buffer[key][n]
                else:
                    traj[key] = self.buffer[key][start_idx:end_idx]
            return traj
    
    def set_traj(self, traj, n):
        # set a specific trajectory in the buffer
        if n >= self.cur_size // self.horizon:
            assert False, "Out of buffer's max index"
        else:
            start_idx = n * self.horizon
            end_idx = start_idx + self.horizon
            for key in traj:
                if 'reset' in key:
                    self.buffer[key][n] = traj[key]
                else:
                    self.buffer[key][start_idx:end_idx] = traj[key]

    # For TD3
    def her_sample(self, batch_size):
        # First randomply select a batch of transitions. Then with probability future_p, the goals will be replaced with the achieved goals
        # and the rewards will be recomputed
        future_p = 1 - (1. / (self.her_args.replay_k + 1))
        her_bool = (np.random.random(batch_size) < future_p).astype(np.int)
        T = self.horizon
        idx = np.random.randint(0, self.cur_size, batch_size)
        traj_idx, traj_t = idx // T, idx % T

        future_idx = batch_rand_int(traj_t, T, batch_size) + traj_idx * T
        next_idx = traj_idx * T + np.minimum(traj_t + 1, T - 1)
        not_done = traj_t < T - 1

        if self.args.input_type == 'pcl':
            dough_pcl, tool_pcl = self.buffer['dough_pcl'][idx], self.buffer['tool_pcl'][idx]
            obs = np.concatenate([dough_pcl, tool_pcl], axis=1)
            goal_obs = self.buffer['goal_pcl'][idx]
            n_dough_pcl, n_tool_pcl = self.buffer['dough_pcl'][next_idx], self.buffer['tool_pcl'][next_idx]
            next_obs = np.concatenate([n_dough_pcl, n_tool_pcl], axis=1)
            action = self.buffer['actions'][idx]
            reward = self.buffer['rewards'][idx].copy()
        else:
            obs = self.buffer['obses'][idx]
            next_obs = self.buffer['obses'][next_idx]
            real_goal_obs, real_goal_pc = self.get_goal(self.buffer['target_v'][idx])
            her_goal_obs = self.buffer['obses'][future_idx]
            goal_obs = (1 - her_bool)[:, None, None, None] * real_goal_obs + her_bool[:, None, None, None] * her_goal_obs
        
            action = self.buffer['actions'][idx]
            # Computing HER reward
            reward = self.buffer['rewards'][idx].copy()
            if len(idx[her_bool > 0]) > 0:
                achieved_state = self.buffer['states'][idx[her_bool > 0]]
                goal_state = self.buffer['states'][future_idx[her_bool > 0]]
                her_reward = self.reward_fn(achieved_state, goal_state)
                reward[her_bool > 0] = her_reward

        return obs, goal_obs, action, next_obs, reward, not_done

    def sample_stacked_obs(self, idx, frame_stack):
        # frame_stack =1 means no stacking
        padded_step = np.concatenate([np.zeros(shape=frame_stack - 1, dtype=np.int), np.arange(self.horizon)])
        traj_idx = idx // self.horizon
        traj_t = idx % self.horizon
        idxes = np.arange(0, frame_stack).reshape(1, -1) + traj_t.reshape(-1, 1)  # TODO For actual stacking, should use negative timestep
        stacked_t = padded_step[idxes]  # B x frame_stack
        stacked_idx = ((traj_idx * self.horizon).reshape(-1, 1) + stacked_t).T  # frame_stack x B
        stack_obs = self.buffer['obses'][stacked_idx]
        stack_obs = np.concatenate(stack_obs, axis=-1)
        return stack_obs

    def load(self, data_path, filename='dataset.gz'):
        if os.path.isfile(data_path):
            # Skip these datasets which have not been finished
            print('Loading dataset from {}'.format(data_path))
            data_path = data_path.replace('pkl', 'gz')
            with gzip.open(data_path, 'rb') as f:
                # self.__dict__ = pickle.load(f) # Does not work well for more datasets
                d = pickle.load(f)

            dataset_buffer = d['buffer']
            N = len(dataset_buffer['obses'])
            if self.cur_pt + N > self.maxlen:
                print('buffer overflows!!!')
                raise NotImplementedError

            for key in dataset_buffer:
                # print(dataset_buffer[key].shape)
                if key == 'mass_grid':
                    print('loading dataset, skipping mass grid')
                    continue
                if key not in self.buffer:
                    if 'reset' in key:
                        self.buffer[key] = np.empty(shape=(self.maxtraj, *dataset_buffer[key].shape[1:]), dtype=dataset_buffer[key].dtype)
                    else:
                        self.buffer[key] = np.empty(shape=(self.maxlen, *dataset_buffer[key].shape[1:]), dtype=dataset_buffer[key].dtype)
                if 'reset' in key:
                    self.buffer[key][self.cur_pt // self.horizon: (self.cur_pt + N) // self.horizon] = dataset_buffer[key]
                else:
                    self.buffer[key][self.cur_pt: self.cur_pt + N] = dataset_buffer[key]
            self.cur_pt += N
            self.cur_size = self.cur_pt

        else:
            datasets = glob(os.path.join(data_path, '**/*dataset*.*z*'), recursive=True)
            for dataset in sorted(datasets):
                self.load(dataset)
            # for exp_folder in sorted(exp_folders):
            #     self.load(os.path.join(exp_folder, filename))
            # dataset_files = glob(os.path.join(data_path, 'dataset*.gz'))
            # print(os.path.join(data_path, 'dataset*.gz'))
            # print(dataset_files)
            # for dataset_file in sorted(dataset_files):
            #     self.load(os.path.join(data_path, dataset_file))

    def save(self, data_path, save_mass_grid=False):
        # https://stackoverflow.com/questions/57983431/whats-the-most-space-efficient-way-to-compress-serialized-python-data
        data_path = data_path.replace('pkl', 'gz')  # gzip compressed file

        d = self.__dict__.copy()  # Shallow copy to avoid large memory usage
        # print(d.keys())

        if self.cur_size < self.maxlen:
            buffer = {}
            for key in self.buffer:
                if 'reset' in key:
                    buffer[key] = self.buffer[key][:self.cur_size // 50]
                else:
                    buffer[key] = self.buffer[key][:self.cur_size]
            d['buffer'] = buffer

        # if not save_mass_grid:
        #     if 'mass_grid' in d['buffer']:
        #         del d['buffer']['mass_grid']
        with gzip.open(data_path, 'wb') as f:
            print('dumping to ', data_path)
            pickle.dump(d, f, protocol=4)

    def __len__(self):
        return self.cur_size
