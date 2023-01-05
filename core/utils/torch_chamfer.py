import torch
def compute_chamfer_distance(x, y):
    """
    Compute the chamfer distance between two point sets.
    :param x: (B, N, 3) torch tensor
    :param y: (B, M, 3) torch tensor
    :return: (B,) torch tensor
    """
    x = x.unsqueeze(2).repeat(1, 1, y.shape[1], 1)
    y = y.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
    dist = torch.sum((x - y) ** 2, dim=-1)
    return torch.mean(torch.min(dist, dim=1)[0] + torch.min(dist, dim=2)[0])
