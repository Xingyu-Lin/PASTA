from copyreg import pickle
import torch
import numpy as np
import torch.nn as nn
from PointFlow.models.networks import PointFlow
import os, json, pickle
from PointFlow.args import get_args
from core.utils.core_utils import VArgs
from core.utils.diffskill_utils import batch_pred_n, batch_pred
from functools import partial
from PointFlow.utils import standard_normal_logprob
from tqdm import tqdm


class PointFlowVAE(object):
    def __init__(self, args, checkpoint_path):
        self.args = args
        print("PointFlowVAE: Resume Path:%s" % checkpoint_path)
        # Load Pointflow Args
        variant_path = os.path.join(os.path.dirname(checkpoint_path), 'variant.json')
        print("PointFlowVAE: loading args from %s" % variant_path)
        with open(variant_path, 'r') as f:
            vv = json.load(f)
        self.pointflow_args = VArgs(vv)
        self.stat = None
        self.model = PointFlow(self.pointflow_args).cuda()
        self.load(checkpoint_path)
        self.cached_buffer_obs = None
        self.cached_buffer_goal = None

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load(self, checkpoint_path):
        # assert not self.pointflow_args.train_T
        # Making sure CNF's time_length is the same as training
        # assert not vv['train_T'] and vv['use_latent_flow']
        # for layer in self.model.latent_cnf.module.chain:
        #     try:
        #         print('Prev latent layer.T:', layer.T)
        #         print('Chainging to ', vv['time_length'])
        #         layer.T = vv['time_length']
        #     except AttributeError:
        #         pass
        # for layer in self.model.point_cnf.module.chain:
        #     try:
        #         print('Prev point layer.T:', layer.T)
        #         print('Chainging to ', vv['time_length'])
        #         layer.T = vv['time_length']
        #     except AttributeError:
        #         pass
        # self.pointflow_args.time_length = vv['time_length']
        def _transform_(m):
            return nn.DataParallel(m)

        self.pointflow_args.train_T = False
        # assert torch.cuda.device_count() == 1, "must load with 1 gpu"
        self.model.multi_gpu_wrapper(_transform_)

        checkpoint = torch.load(checkpoint_path)
        # Loading Model
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        # Loading Training data stats
        self.load_stat(os.path.join(os.path.dirname(checkpoint_path), 'train_stat.pkl'))
        # Reset cached encoding
        self.cached_buffer_obs = None
        self.cached_buffer_goal = None

    def load_stat(self, stat_path):
        # Load training dataset statistics
        with open(stat_path, 'rb') as f:
            self.stat = pickle.load(f)
        print("Loading stats")
        for k, v in self.stat.items():
            self.stat[k] = torch.from_numpy(np.array(v, dtype=np.float32)).cuda()

    def reconstruct(self, x, num_points=None, truncate_std=None):
        """x: point cloud, B x N x 3"""
        # 1. normalize x; 2. reconstruct; 3. re-normalize reconstructed x
        assert self.stat is not None
        t = torch.mean(x, dim=1, keepdim=True)
        x = (x - t) / self.stat['std']
        out = self.model.reconstruct(x, num_points=num_points, truncate_std=truncate_std)
        return out * self.stat['std'] + t

    def encode(self, x):
        # normalize x
        # return both z and t, t = torch.mean(x, dim=1)
        # POTENTIALLY NEED TO ALSO ADD THE GLBOAL STD
        assert self.stat is not None
        t = torch.mean(x, dim=1, keepdim=True)  # B x 1 x 3
        x = (x - t) / self.stat['std']
        return self.model.encode(x), t[:, 0, :]

    def encode_u(self, x):
        z, t = self.encode(x)
        return torch.cat([z, t], dim=1)

    def decode(self, u, num_points, truncate_std=None):
        assert self.stat is not None
        z, t = u[:, :-3], u[:, -3:]
        noise, out = self.model.decode(z, num_points, truncate_std=truncate_std)
        return out * self.stat['std'] + t.view(z.shape[0], 1, 3)

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None):
        # sample t, apply std and return the unnormalized point cloud
        assert self.stat is not None
        t_sample = self.sample_t(batch_size)
        _, x = self.model.sample(batch_size, num_points, truncate_std=truncate_std, truncate_std_latent=truncate_std_latent, gpu='cuda')
        x = x * self.stat['std'] + t_sample
        return x

    def sample_latent(self, batch_size, truncate_std_latent=None, backtocpu=False):
        ret = self.model.sample_latent(batch_size, truncate_std_latent=truncate_std_latent, gpu='cuda')
        if backtocpu:
            return ret.cpu()
        return ret

    def generate_cached_z(self, B=10000000):
        if self.args.debug:
            B = 100000
        self.cached_z = batch_pred_n(partial(self.sample_latent, backtocpu=True), N=B, batch_size=10000)

    def generate_cached_buffer(self, buffer):
        def encode(x):
            return self.encode_u(x.cuda()).detach().cpu()

        N = len(buffer)
        # Get states
        dpc, _ = buffer.get_state(np.arange(N))
        self.cached_buffer_obs = batch_pred(encode, {'x': torch.FloatTensor(dpc)}, batch_size=128)
        self.cached_buffer_goal = batch_pred(encode, {'x': torch.FloatTensor(buffer.np_target_pc)}, batch_size=128)
        print('Cached buffer of length {} generated!'.format(N))

    def generate_cached_set_buffer(self, buffer):
        from core.utils.pc_utils import decompose_pc
        from core.utils.core_utils import array_to_list, list_to_array
        num_points = 1000

        def encode(x):
            return self.encode_u(x.cuda()).detach().cpu()

        N = len(buffer)

        # Get states
        dpc, _ = buffer.get_state(np.arange(N))
        labels = buffer.buffer['dbscan_labels']
        all_pcs = []
        for i, (pc, label) in tqdm(enumerate(zip(dpc, labels)), desc='Decompose pc'):
            pcs = decompose_pc(pc, label, N=num_points)
            all_pcs.append(np.array(pcs))
        self.all_pcs = all_pcs
        all_pcs, obs_idx = list_to_array(all_pcs)

        goal_pcs = []
        goal_labels = buffer.np_target_dbscan
        for i, (pc, label) in tqdm(enumerate(zip(buffer.np_target_pc, goal_labels)), desc='Decompose goal pc'):
            pcs = decompose_pc(pc, label, N=num_points)
            goal_pcs.append(np.array(pcs))
        self.goal_pcs = goal_pcs
        goal_pcs, goal_idx = list_to_array(goal_pcs)

        u_obs = batch_pred(encode, {'x': torch.FloatTensor(all_pcs)}, batch_size=128).numpy()
        u_goal = batch_pred(encode, {'x': torch.FloatTensor(goal_pcs)}, batch_size=128).numpy()
        self.cached_buffer_obs = array_to_list(u_obs, obs_idx)
        self.cached_buffer_goal = array_to_list(u_goal, goal_idx)
        print('Cached buffer of length {} generated!'.format(N))

    def get_cached_encoding(self, idx, noise=False, goal=False):
        if goal:
            u = self.cached_buffer_goal[idx]
        else:
            u = self.cached_buffer_obs[idx]
        if not noise:
            return u
        noise = (torch.rand(u.shape, device=u.device) - 0.5) * 2 * self.args.fea_z_noise
        noise[:, -3:] = (torch.rand(noise[:, -3:].shape, device=u.device) - 0.5) * 2 * self.args.fea_t_noise
        return u + noise

    def sample_u(self, batch_size, truncate_std_latent=None, cache=False):
        """Sample latent as well as the translation"""
        if cache:
            assert truncate_std_latent is None
            if not hasattr(self, 'cached_z'):
                self.generate_cached_z()
            idx = torch.randint(0, len(self.cached_z), (batch_size,))
            z = self.cached_z[idx].cuda()
            z += torch.normal(0, 0.1, size=z.shape, device=z.device)
        else:
            z = self.model.sample_latent(batch_size, truncate_std_latent=truncate_std_latent, gpu='cuda')
        trans = self.sample_t(batch_size)[:, 0, :]
        return torch.cat([z, trans], dim=-1)

    def sample_t(self, batch_size, t_min=None, t_max=None):
        """ Return B x 1 x 3"""
        if t_min is None or t_max is None:
            assert self.stat is not None
            t_min = torch.min(self.stat['mean'], dim=0, keepdim=True)[0]
            t_max = torch.max(self.stat['mean'], dim=0, keepdim=True)[0]
        # t_max[0, 1] = 0.08
        t_sample = torch.distributions.Uniform(t_min, t_max).sample((batch_size,))
        return t_sample
    
    def sample_unif_z(self, batch_size):
        if not hasattr(self, 'cached_z'):
            self.generate_cached_z()
        if not hasattr(self, 'z_min'):
            self.z_min, self.z_max = torch.min(self.cached_z, dim=0)[0].cuda(), torch.max(self.cached_z, dim=0)[0].cuda()
        zs = torch.distributions.Uniform(self.z_min, self.z_max).sample((batch_size,))
        return zs

    def calc_pz(self, z):
        oriB = len(z)
        rep = 5
        z = z[None].repeat(rep, 1, 1).view(rep * oriB, -1)
        B = len(z)
        w, delta_log_pw = self.model.latent_cnf(z, None, torch.zeros(B, 1).to(z))
        log_pw = standard_normal_logprob(w).view(B, -1).sum(1, keepdim=True)
        delta_log_pw = delta_log_pw.view(B, 1)
        log_pz = log_pw - delta_log_pw
        log_pz = log_pz.view(rep, oriB).mean(dim=0)
        return log_pz
