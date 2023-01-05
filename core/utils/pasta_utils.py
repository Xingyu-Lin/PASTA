import numpy as np
import torch

from core.utils.pc_utils import resample_pc
from core.utils.torch_chamfer import compute_chamfer_distance

# Define the cardinality of each skill
cut = {'in': 1, 'out': 2}
rearrange = {'in': 1, 'out': 1}
spread = {'in': 1, 'out': 1}

cut_unfilter = {'in': 1, 'out': 2}
rearrange_unfilter = {'in': 2, 'out': 2}

env_skills = \
    {'CutRearrange-v1': [cut, rearrange],
     'CutRearrangeSpread-v1': [cut, rearrange, spread],
     'CutRearrangeSpread-v2': [cut, rearrange, spread]}

env_skills_unfilter = \
    {'CutRearrange-v1': [cut_unfilter, rearrange_unfilter],
    'CutRearrangeSpread-v1': [cut_unfilter, rearrange, spread]}

def get_skill_in_out(env_name, filter_set=True):
    if filter_set:
        skills = env_skills[env_name]
        sin = np.array([skill['in'] for skill in skills])
        sout = np.array([skill['out'] for skill in skills])
    else:
        skills = env_skills_unfilter[env_name]
        sin = np.array([skill['in'] for skill in skills])
        sout = np.array([skill['out'] for skill in skills])
    return sin, sout

def generate_cached_grid_idx():
    cached_grids = {}
    for c1 in range(10):
        for c2 in range(10):
            l = []
            for i in range(c1):
                for j in range(c2):
                    l.append([i, j])
            cached_grids[(c1, c2)] = np.array(l)
    return cached_grids


cached_grids = generate_cached_grid_idx()


def match_set(pcs_obs, pcs_goal, us_obs, us_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200, verbose=False, thr=5e-4):
    """ Given idx of obs and goal, match the set representation based on the point cloud and return them """

    def dump():
        import pickle
        test_path = './data/test_data/match_set.pkl'
        d = {'pcs_obs': pcs_obs, 'pcs_goal': pcs_goal, 'us_obs': us_obs, 'us_goal': us_goal, 'n_in': n_in, 'n_out': n_out}
        with open(test_path, 'wb') as f:
            pickle.dump(d, f)
        print('Saving test data to ', test_path)

    if verbose:  # For debugging
        dump()
    stacked_pc1, stacked_pc2, node_idx1, node_idx2 = [], [], [], []
    downsample_idx = np.random.choice(len(pcs_obs[0][0]), chamfer_downsample)
    cnt1, cnt2 = 0, 0
    for pcs1, pcs2 in zip(pcs_obs, pcs_goal):
        pair_idx = cached_grids[(len(pcs1), len(pcs2))]  # N x 2
        stacked_pc1.append(np.array(pcs1)[pair_idx[:, 0]][:, downsample_idx])
        stacked_pc2.append(np.array(pcs2)[pair_idx[:, 1]][:, downsample_idx])
        node_idx1.append((pair_idx[:, 0] + cnt1).reshape(-1, 1))
        node_idx2.append((pair_idx[:, 1] + cnt2).reshape(-1, 1))
        cnt1 += len(pcs1)
        cnt2 += len(pcs2)
    stacked_pc1, stacked_pc2 = np.vstack(stacked_pc1), np.vstack(stacked_pc2)
    node_idx1, node_idx2 = np.vstack(node_idx1)[:, 0], np.vstack(node_idx2)[:, 0]

    if dist == 'chamfer':
        dist = compute_chamfer_distance(torch.FloatTensor(stacked_pc1).cuda(), torch.FloatTensor(stacked_pc2).cuda())
        dist = dist.detach().cpu().numpy()
        if verbose:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.hist(np.log(dist + 1e-6), bins=200)
            plt.axvline(x=np.log(thr + 1e-6), color='r')
            plt.tight_layout()
            plt.savefig('test_match/dist_hist.png')
    else:
        raise NotImplementedError

    match = np.where(dist < thr)
    mask1, mask2 = np.ones(cnt1, dtype=np.int), np.ones(cnt2, dtype=np.int)
    mask1[node_idx1[match]], mask2[node_idx2[match]] = 0, 0  # Scatter

    cnt1, cnt2 = 0, 0
    all_u_obs_goal, pc_idx, pc_idx1, pc_idx2 = [], [], [], []
    for i, (pcs1, pcs2, u1, u2) in enumerate(zip(pcs_obs, pcs_goal, us_obs, us_goal)):
        # print('in and out:', sum(mask1[cnt1:cnt1 + len(pcs1)]), sum(mask2[cnt2:cnt2 + len(pcs2)]))
        if sum(mask1[cnt1:cnt1 + len(pcs1)]) == n_in and sum(mask2[cnt2:cnt2 + len(pcs2)]) == n_out:  # Passed filter
            m1, m2 = np.where(mask1[cnt1:cnt1 + len(pcs1)]), np.where(mask2[cnt2:cnt2 + len(pcs2)])
            u_obs = u1[m1]
            u_goal = u2[m2]
            u_obs_goal = np.concatenate([u_obs, u_goal]).flatten()
            all_u_obs_goal.append(u_obs_goal)
            pc_idx.append(i)
            pc_idx1.append(m1)
            pc_idx2.append(m2)
        cnt1 += len(pcs1)
        cnt2 += len(pcs2)
    return np.vstack(all_u_obs_goal), {'pc_idx': pc_idx, 'pc_idx1': pc_idx1, 'pc_idx2': pc_idx2}

def match_set_pcl(n_sample, pcs_obs, pcs_goal, n_in, n_out, dist='chamfer', chamfer_downsample=200, verbose=False, thr=5e-4, eval=False):
    """ Given idx of obs and goal, match the set representation based on the point cloud and return them
        Note that this is only used for the policy, so when pcs_obs have the same number of components as n_out, 
        i.e. sum(mask1[cnt1:cnt1 + len(pcs1)]) == n_out, or vice versa, it also gets added to valid observation.
    """

    def dump():
        import pickle
        test_path = './data/test_data/match_set.pkl'
        d = {'pcs_obs': pcs_obs, 'pcs_goal': pcs_goal, 'n_in': n_in, 'n_out': n_out}
        with open(test_path, 'wb') as f:
            pickle.dump(d, f)
        print('Saving test data to ', test_path)

    if verbose:  # For debugging
        dump()
    stacked_pc1, stacked_pc2, node_idx1, node_idx2 = [], [], [], []
    downsample_idx = np.random.choice(len(pcs_obs[0][0]), chamfer_downsample)
    cnt1, cnt2 = 0, 0
    for pcs1, pcs2 in zip(pcs_obs, pcs_goal):
        pair_idx = cached_grids[(len(pcs1), len(pcs2))]  # N x 2
        stacked_pc1.append(np.array(pcs1)[pair_idx[:, 0]][:, downsample_idx])
        stacked_pc2.append(np.array(pcs2)[pair_idx[:, 1]][:, downsample_idx])
        node_idx1.append((pair_idx[:, 0] + cnt1).reshape(-1, 1))
        node_idx2.append((pair_idx[:, 1] + cnt2).reshape(-1, 1))
        cnt1 += len(pcs1)
        cnt2 += len(pcs2)
    stacked_pc1, stacked_pc2 = np.vstack(stacked_pc1), np.vstack(stacked_pc2)
    node_idx1, node_idx2 = np.vstack(node_idx1)[:, 0], np.vstack(node_idx2)[:, 0]

    if dist == 'chamfer':
        dist = compute_chamfer_distance(torch.FloatTensor(stacked_pc1).cuda(), torch.FloatTensor(stacked_pc2).cuda())
        dist = dist.detach().cpu().numpy()
        if verbose:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.hist(np.log(dist + 1e-6), bins=200)
            plt.axvline(x=np.log(thr + 1e-6), color='r')
            plt.tight_layout()
            plt.savefig('test_match/dist_hist.png')
    else:
        raise NotImplementedError
    match = np.where(dist < thr)
    mask1, mask2 = np.ones(cnt1, dtype=np.int), np.ones(cnt2, dtype=np.int)
    mask1[node_idx1[match]], mask2[node_idx2[match]] = 0, 0  # Scatter

    cnt1, cnt2 = 0, 0
    all_obs, all_goals, pc_idx, pc_idx1, pc_idx2 = [], [], [], [], []
    for i, (pcs1, pcs2) in enumerate(zip(pcs_obs, pcs_goal)):
        if sum(mask1[cnt1:cnt1 + len(pcs1)]) in [n_in, n_out] and sum(mask2[cnt2:cnt2 + len(pcs2)]) in [n_in, n_out]:  # Passed filter
            m1, m2 = np.where(mask1[cnt1:cnt1 + len(pcs1)]), np.where(mask2[cnt2:cnt2 + len(pcs2)])
            pc1 = np.array(pcs1)[m1].reshape(-1, 3)
            pc2 = np.array(pcs2)[m2].reshape(-1, 3)
            pc1 = resample_pc(pc1, 1000)
            pc2 = resample_pc(pc2, 1000)
            all_obs.append(pc1)
            all_goals.append(pc2)
            pc_idx.append(i)
            pc_idx1.append(m1)
            pc_idx2.append(m2)
        cnt1 += len(pcs1)
        cnt2 += len(pcs2)
        if len(all_obs) == n_sample:
            break
    if len(all_obs) == 0:
        assert eval
        if len(match[0]) == 0:   # everything is above thr, rematch
            print(match)
            print("Rematching with thr:", np.min(dist)+1e-6)
            return match_set_pcl(n_sample, pcs_obs, pcs_goal, n_in, n_out, dist='chamfer', 
                chamfer_downsample=chamfer_downsample, thr=np.min(dist)+1e-6, eval=eval)
        elif n_in != n_out: # cut
            return None, None, {}
        else:   # too many things get filtered, find the pair with max_dist and return. only works when 1:2 or 2:1
            pc_idx, pc_idx1, pc_idx2 = [], [], []
            idx_max = np.argmax(dist)
            m1 = (np.array([node_idx1[idx_max]]), )
            m2 = (np.array([node_idx2[idx_max]]), )
            pc1 = np.array(pcs1)[m1].reshape(-1, 3)
            pc2 = np.array(pcs2)[m2].reshape(-1, 3)
            pc1 = resample_pc(pc1, 1000)
            pc2 = resample_pc(pc2, 1000)
            all_obs.append(pc1)
            all_goals.append(pc2)
            pc_idx.append(0)
            pc_idx1.append(m1)
            pc_idx2.append(m2)
            return np.vstack(all_obs).reshape(-1, 1000, 3), np.vstack(all_goals).reshape(-1, 1000, 3), {'pc_idx': pc_idx, 'pc_idx1': pc_idx1, 'pc_idx2': pc_idx2}
        
    return np.vstack(all_obs).reshape(-1, 1000, 3), np.vstack(all_goals).reshape(-1, 1000, 3), {'pc_idx': pc_idx, 'pc_idx1': pc_idx1, 'pc_idx2': pc_idx2}

