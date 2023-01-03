import numpy as np


def batch_resample_pc(pcs, N):
    """ Resample the pointcloud to have N points. pcs has shape B x M x 3
        Return pc shape: B x N x 3
    """
    B, M = pcs.shape[:2]
    if M == N:
        return pcs
    if M < N:
        K = N - M
        idx = np.random.choice(M, K)
        return np.concatenate([pcs, pcs[:, idx]], axis=1)
    else:
        idx = np.random.choice(M, N, replace=False)
        return pcs[:, idx]


if __name__ == '__main__':
    import numpy as np

    a = np.random.random([128, 500, 3])
    print(batch_resample_pc(a, 1000).shape)


def resample_pc(pc, N):
    """Resample the pointcloud to have N points"""
    if len(pc) == N:
        return pc.reshape(N, 3)
    if len(pc) < N:
        m = N - len(pc)
        idx = np.random.choice(len(pc), m)
        return np.vstack([pc, pc[idx]])
    else:
        idx = np.random.choice(len(pc), N)
        return pc[idx].reshape(N, 3)


def decompose_pc(pc, label, N=None):
    """Decompose a pointcloud into set based on dbscan label.
        If N is not None, resample each pc to have N points
    """
    assert len(pc) == len(label)
    max_label = np.max(label)
    pcs = []
    for l in range(max_label + 1):
        p = pc[np.where(label == l)]
        p = resample_pc(p, N)
        pcs.append(p)
    return pcs


def combine_pc_list(list_pcs):
    pcs = []
    for i in range(len(list_pcs)):
        pcs.append(np.vstack(list_pcs[i]))
    return pcs
