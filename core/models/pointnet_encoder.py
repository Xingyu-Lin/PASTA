# import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, global_mean_pool
from torch_geometric.nn import knn_interpolate


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        # print(idx.shape, edge_index.shape)
        # print("avg # pt per centroid:", edge_index.shape[1] // idx.shape[0])
        if x is None:
            x = self.conv((x, x), (pos, pos[idx]), edge_index)
        else:
            x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn
        self.argmax_points = torch.zeros(1)

    def forward(self, x, pos, batch, calc_argmax=False):
        if x is not None:
            input = self.nn(torch.cat([x, pos], dim=1))
        else:
            input = self.nn(pos)
        x = global_max_pool(input, batch)
        if calc_argmax:
            self.argmax_idx = torch.argmax(input, dim=0)
            self.argmax_points = pos[self.argmax_idx]
        # x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class PointNetEncoder(torch.nn.Module):
    """PointNet++"""
    def __init__(self, in_dim):
        super(PointNetEncoder, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.05, MLP([3 + in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.1, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 128, 512, 1024]))

    def forward(self, data, detach=False):
        sa0_out = (data['x'], data['pos'], data['batch'])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        if detach:
            x.detach()
        return x

class PointNetEncoderCat(torch.nn.Module):
    """PointNet++"""
    def __init__(self, in_dim):
        super(PointNetEncoderCat, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.05, MLP([3 + in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.1, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 128, 512, 1024]))

    def forward(self, data, detach=False):
        sa0_out = (data['x'], data['pos'], data['batch'])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        if detach:
            x.detach()
        return x


class PointNetEncoder2(torch.nn.Module):
    """PointNet"""
    def __init__(self, in_dim):
        super(PointNetEncoder2, self).__init__()
        # Input channels account for both `pos` and node features.
        self.sa3_module = GlobalSAModule(MLP([in_dim, 128, 128, 256]))

    def forward(self, data, detach=False, calc_argmax=False):
        sa3_out = self.sa3_module(*data, calc_argmax=calc_argmax)
        x, pos, batch = sa3_out
        if detach:
            x.detach()
        return x


class PointNetEncoder3(torch.nn.Module):
    """PointNet++, pointwise prediction"""
    def __init__(self, in_dim):
        super(PointNetEncoder3, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.05, MLP([3 + in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.1, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 3)

    def forward(self, data, detach=False):
        sa0_out = (data['x'], data['pos'], data['batch'])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = self.lin1(x)

        if detach:
            x.detach()
        return x
