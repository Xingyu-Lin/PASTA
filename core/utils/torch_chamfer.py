import torch

def compute_chamfer_distance(x, y):
    """
    Compute the chamfer distance loss between two point sets.
    :param x: (B, N, 3) torch tensor
    :param y: (B, M, 3) torch tensor
    :return: (B,) torch tensor
    """
    x = x.unsqueeze(2).repeat(1, 1, y.shape[1], 1)
    y = y.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
    dist = torch.sum((x - y) ** 2, dim=-1)
    dist1, _ = torch.min(dist, dim=1)
    dist2, _ = torch.min(dist, dim=2)
    return dist1.mean(dim=1) + dist2.mean(dim=1)

if __name__ == '__main__':
    x = torch.rand(512, 500, 3)
    y = torch.rand(512, 200, 3)
    print(compute_chamfer_distance(x, y).shape)
