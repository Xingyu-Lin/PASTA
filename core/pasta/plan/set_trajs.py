import numpy as np
import torch
from core.utils.pc_utils import batch_resample_pc
cached_perm = {}


def batch_random_choice(N, D, batch_size):
    # Randomly select D numbers from [0, N-1].
    # Return the selected number as well as the un-selected
    if N not in cached_perm:
        from itertools import permutations
        cached_perm[N] = np.array(list(permutations(range(N))))
    perm = cached_perm[N]
    idx = np.random.randint(0, len(perm), batch_size)
    return perm[idx][:, :D].copy().reshape(batch_size, D), perm[idx][:, D:].copy().reshape(batch_size, N - D)


def select(u, select_idx):
    # u: B x K x D
    # select_idx: B x K2
    # Return: Select K2 from K and return B x K2 x D
    B, K2 = select_idx.shape
    D = u.shape[2]
    batch_idx = np.tile(np.arange(B).reshape(B, 1), [1, K2]).flatten()
    return u[batch_idx.flatten(), select_idx.flatten()].view(B, K2, D)

def select_his(u, select_idx):
    # u: T x B x K x D
    # select_idx: B x K2
    # Return: Select K2 from K and return T x B x K2 x D
    B, K2 = select_idx.shape
    D = u.shape[-1]
    T = u.shape[0]
    batch_idx = np.tile(np.arange(B).reshape(B, 1), [1, K2]).flatten()
    return u[:, batch_idx.flatten(), select_idx.flatten()].reshape(T, B, K2, D)


def u_obs_goal_to_pc(agent, u_obs_goal, n_in, n_out, num_points=1000):
    B, dimu = u_obs_goal.shape[0], u_obs_goal.shape[1] // (n_in + n_out)
    with torch.no_grad():
        pcs = agent.vae.decode(u_obs_goal.view(-1, dimu), num_points).view(B, n_in + n_out, num_points, 3)
    pcs_obs = pcs[:, :n_in, :, :].reshape(B, n_in * num_points, 3)
    pcs_obs = batch_resample_pc(pcs_obs, num_points)
    pcs_goal = pcs[:, n_in:, :, :].reshape(B, n_out * num_points, 3)
    pcs_goal = batch_resample_pc(pcs_goal, num_points)
    return pcs_obs.detach().cpu().numpy(), pcs_goal.detach().cpu().numpy()


def struct_u_to_pc(agent, struct_u, num_points=1000, tensor=False, in_tensor=True):
    from core.utils.core_utils import array_to_list, list_to_array
    from core.utils.pc_utils import resample_pc
    arr_u, list_idx = list_to_array(struct_u, tensor=in_tensor, axis=1)
    if not in_tensor:
        arr_u = torch.FloatTensor(arr_u).to(agent.device)
    with torch.no_grad():
        B, K, D = arr_u.shape
        arr_u = arr_u.view(B * K, D)
        pcs = agent.vae.decode(arr_u, num_points)
        pcs = pcs.view([B, K, num_points, 3])  # K = tot[0] + tot[1] + tot[2]
    if not tensor:
        pcs = pcs.detach().cpu().numpy()
        list_pcs = array_to_list(pcs, list_idx, axis=1)
        uniform_pcs = []
        for pcs in list_pcs:  # For each time step
            for i in range(pcs.shape[0]):
                cat_pc = np.vstack(pcs[i])
                pc = resample_pc(cat_pc, num_points)
                uniform_pcs.append(pc)
        uniform_pcs = np.array(uniform_pcs).reshape(-1, B, num_points, 3).swapaxes(0, 1)
        return uniform_pcs, list_pcs  # B x H x pc_shape
    else:
        raise NotImplementedError


class SetTrajs(object):
    def __init__(self, tids, sin, sout, tiled_u_obs, tiled_u_goal, sample_func):
        self.tids = tids
        self.sin = sin
        self.sout = sout
        self.tiled_u_obs = tiled_u_obs.detach()
        self.tiled_u_goal = tiled_u_goal.detach()
        self.B, self.K0, self.D = tiled_u_obs.shape
        self.u_opt_idxes = []
        # tots: Total number of entities
        # masks_in: Attention in for the skill
        # masks_keep: Complement of the attention
        # opt_idxes: Index of the sub-goals to be optimized in the u_samples
        self.tots, self.masks_in, self.masks_keep, self.u_opt_idxes = [], [], [], []
        self.tots.append(tiled_u_obs.shape[1])

        cnt = 0  # Number of u samples needed to initialize intermediate sub-goals

        for step, tid in enumerate(tids):
            n_in, n_out = sin[tid], sout[tid]
            u_opt_idx = np.array(list(range(cnt, cnt + self.B * n_out)))
            cnt += self.B * n_out
            self.u_opt_idxes.append(u_opt_idx)
            self.tots.append(self.tots[-1] + n_out - n_in)
            mask_in, mask_keep = batch_random_choice(self.tots[-2], n_in, self.B)
            self.masks_in.append(mask_in)
            self.masks_keep.append(mask_keep)

        # u_samples = sample_func(sum(self.tots[1:]) * self.B).detach()
        u_samples = sample_func(np.max(u_opt_idx) + 1).detach()
        arr_u_opt_idxes = np.concatenate(self.u_opt_idxes, axis=0).flatten()
        self.all_u_opt = u_samples[arr_u_opt_idxes]  # All u samples that can be optimized
        self.struct_u = None

    def set_sol(self, sol_u, sol_u_his=None):
        assert self.all_u_opt.shape == sol_u.shape
        self.all_u_opt = sol_u
        self.struct_u, _ = self.get_structured_u(sol_u)
        if sol_u_his is not None:
            self.struct_u_his = self.get_structured_u_his(sol_u_his)

    def get_structured_u(self, all_u_opt=None):
        """
        all_u_opt: N x D, all the sub-goals to be optimized
        self.u_opt_idxes: List of the indexes for the optimized variables for each step, each of the shape B * n_out
        self.masks: masks_in and masks_keep, the attention mask for each step
        return:
            all_u: List of batched u for each step including the non-optimized u
            all_obs_goal: List of concatenated obs_goal for each step
        """

        all_u_opt = self.all_u_opt if all_u_opt is None else all_u_opt
        curr_u = self.tiled_u_obs
        all_struct_u, all_obs_goal = [], []
        for step, u_opt_idx in enumerate(self.u_opt_idxes):
            mask_in, mask_keep = self.masks_in[step], self.masks_keep[step]  # B x n_in
            u_opt = all_u_opt[u_opt_idx].view(self.B, -1, self.D)
            u_keep = select(curr_u, mask_keep)
            obs_goal = torch.cat([select(curr_u, mask_in), u_opt], dim=1).view(self.B, -1)
            curr_u = torch.cat([u_opt, u_keep], dim=1)  # Always put the opt vars on the top in the output
            all_struct_u.append(curr_u)
            all_obs_goal.append(obs_goal)
        return all_struct_u, all_obs_goal
        
    def get_structured_u_his(self, sol_u_his):
        """
        sol_u_his: Iteration x N x D, all history the sub-goals to be optimized
        self.u_opt_idxes: List of the indexes for the optimized variables for each step, each of the shape B * n_out
        self.masks: masks_in and masks_keep, the attention mask for each step
        return:
            all_struct_u_his: List of batched u_his for each step including the non-optimized u
        """
        T = sol_u_his.shape[0]
        curr_u = np.tile(self.tiled_u_obs.detach().cpu().numpy(), (T, 1, 1, 1))
        all_struct_u_his = []
        for step, u_opt_idx in enumerate(self.u_opt_idxes):
            mask_in, mask_keep = self.masks_in[step], self.masks_keep[step]  # B x n_in
            u_opt = sol_u_his[:, u_opt_idx].reshape(T, self.B, -1, self.D)
            u_keep = select_his(curr_u, mask_keep)
            curr_u = np.concatenate([u_opt, u_keep], axis=2)  # Always put the opt vars on the top in the output
            all_struct_u_his.append(curr_u)
        return all_struct_u_his

    def select(self, idxes):
        """Select some of the struct u"""
        sel_struct_u = []
        idxes = np.array(idxes)  # Need this for some reason...
        for u in self.struct_u:
            sel_struct_u.append(u.clone()[idxes])
        return sel_struct_u

    def select_his(self, idxes):
        """Select some of the struct u_his"""
        assert self.struct_u_his is not None
        sel_struct_u_his = []
        idxes = np.array(idxes)  # Need this for some reason...
        for u_his in self.struct_u_his:
            sel_struct_u_his.append(u_his.copy()[:, idxes])
        return sel_struct_u_his
