import torch
from torch import nn
import numpy as np
from lxyTools.pytorchTools import Attention

class DCNN(nn.Module):

    '''
    arg:
        edge_index: 同torch_geometric里面的edge_index相同
        k: 最多考虑多少步，即转移矩阵的幂级数的长度
    '''

    def __init__(self, edge_index, k, input_dim):
        nn.Module.__init__(self)
        num_nodes = edge_index.max() + 1

        a = np.zeros((num_nodes, num_nodes))

        for i in range(len(edge_index[0])):
             a[edge_index[0][i], edge_index[1][i]] = 1
        a = a/(a.sum(axis=1).reshape(-1, 1))

        P = np.zeros((num_nodes, k, num_nodes))

        power_val = np.eye(num_nodes)
        for i in range(P.shape[1]):
            P[:, i, :] = power_val
            power_val = np.matmul(power_val, a)

        Pt = torch.tensor(P, dtype=torch.float, requires_grad=False)

        Wc = torch.randn(k, input_dim, dtype=torch.float, requires_grad=True)/10

        self.Pt = nn.Parameter(Pt, requires_grad=False)
        self.Wc = nn.Parameter(Wc, requires_grad=True)

    def forward(self, x):
        return torch.mul(torch.matmul(self.Pt, x), self.Wc)


class attentionDCNN(nn.Module):
    '''
    arg:


    '''
    def __init__(self, edge_index, k, input_dim):
        nn.Module.__init__(self)
        num_nodes = edge_index.max() + 1

        a = np.zeros((num_nodes, num_nodes))

        for i in range(len(edge_index[0])):
             a[edge_index[0][i], edge_index[1][i]] = 1
        a = a/(a.sum(axis=1).reshape(-1, 1))

        P = np.zeros((num_nodes, k, num_nodes))

        power_val = np.eye(num_nodes)
        for i in range(P.shape[1]):
            P[:, i, :] = power_val
            power_val = np.matmul(power_val, a)

        Pt = torch.tensor(P, dtype=torch.float, requires_grad=False)

        self.Pt = nn.Parameter(Pt, requires_grad=False)

        self.atten = Attention(input_dim, k)

    def forward(self, x):
        x = torch.matmul(self.Pt, x)
        re = self.atten(x)
        return re
