import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import math
import os
import random

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm
import visdom


def set_random_seed(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GatedConv1d(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.conv1d = nn.Conv1d(**kwargs)
        self.sigmoid = nn.Sigmoid()

        assert kwargs['out_channels']%2==0
        self.out_channels = int(kwargs['out_channels']/2)

        self.linear = nn.Linear(self.out_channels, 1)


    def forward(self, x):

        conv1d_out = self.conv1d(x)
        P = conv1d_out[:, :self.out_channels, :]
        Q = conv1d_out[:, self.out_channels:, :]

        Q = self.linear(Q.contiguous().transpose(1, 2)).contiguous().transpose(1, 2).contiguous()

        Q = self.sigmoid(Q)

        out = P*Q

        return out






# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         nn.Module.__init__(self)
#
#         self.hidden_size = hidden_size
#
#         self.linear_ir = nn.Linear(input_size, hidden_size)
#         self.linear_hr = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_iz = nn.Linear(input_size, hidden_size)
#         self.linear_hz = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_in = nn.Linear(input_size, hidden_size)
#         self.linear_hn = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_out = nn.Linear(hidden_size, hidden_size)
#
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#     def step(self, x, h):
#
#         r = self.sigmoid(self.linear_ir(x) + self.linear_hr(h))
#         z = self.sigmoid(self.linear_iz(x) + self.linear_hz(h))
#         n = self.tanh(self.linear_in(x) + torch.mul(r, self.linear_hn(h)))
#
#         h_new = torch.mul((1-z), n) + torch.mul(z, h)
#
#         y = self.linear_out(h_new)
#         return y, h_new
#
#     def forward(self, x):
#
#         outlist = []
#         h = torch.tensor(np.zeros((x.shape[0], self.hidden_size)), dtype=torch.float32).cuda()
#         for i in range(x.shape[1]):
#
#             y,h = self.step(x[:, i, :], h)
#             y = y.contiguous().view(-1, 1, self.hidden_size)
#             outlist.append(y)
#         return torch.cat(outlist, dim=1)
#
# class BiGRU(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         nn.Module.__init__(self)
#         self.gru1 = GRU(input_size, hidden_size)
#         self.gru2 = GRU(input_size, hidden_size)
#     def forward(self, x):
#         x_flip = torch.flip(x, [1])
#
#         y1 = self.gru1(x)
#         y2 = self.gru1(x_flip)
#
#         return torch.cat([y1, y2], 2)



class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class selfAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, input_dim, dk=None):
        nn.Module.__init__(self)

        self.linear_q = nn.Linear(input_dim, qk_dim)
        self.linear_k = nn.Linear(input_dim, qk_dim)
        self.linear_v = nn.Linear(input_dim, v_dim)

        if dk is not None:
            self.dk = dk
        else:
            self.dk = qk_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        attention = torch.matmul(q, k.transpose(1, 2))/math.sqrt(self.dk)
        attention = self.softmax(attention)
        context = torch.matmul(attention, v)

        return context


class multilayerSelfAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, input_dim, num_layers=1, dk=None):
        nn.Module.__init__(self)
        self.attentionlist = nn.ModuleList([selfAttention(qk_dim, v_dim, input_dim, dk) for i in range(num_layers)])

    def forward(self, x):
        out = x
        for atten in self.attentionlist:
            out = atten(out)
        return out




class multiHeadAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, input_dim, h, dk=None):
        nn.Module.__init__(self)

        self.linear_q = nn.Linear(input_dim, qk_dim)
        self.linear_k = nn.Linear(input_dim, qk_dim)
        self.linear_v = nn.Linear(input_dim, v_dim)

        self.head_num = h

        if dk is not None:
            self.dk = dk
        else:
            self.dk = int(qk_dim/h)

        self.dv = int(v_dim/h)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = torch.cat(torch.chunk(q, self.head_num, dim=2), dim=0)
        k = torch.cat(torch.chunk(k, self.head_num, dim=2), dim=0)
        v = torch.cat(torch.chunk(v, self.head_num, dim=2), dim=0)

        attention = torch.matmul(q, k.transpose(1, 2))/math.sqrt(self.dk)
        attention = self.softmax(attention)
        context = torch.matmul(attention, v)

        context = torch.cat(torch.chunk(context, self.head_num, dim=0), dim=2)

        return context


class FM(nn.Module):
    def __init__(self, n, k, bias=True):
        nn.Module.__init__(self)
        self.n = n
        self.k = k

        self.linear = nn.Linear(n, 1, bias=bias)

        self.v = torch.empty(k, n)
        nn.init.xavier_uniform_(self.v)
        self.v = nn.Parameter(self.v)

    def forward(self, x):
        linear_part = self.linear(x)

        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)

        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2, dim=1, keepdim=True)

        return output

class BiInteraction(nn.Module):
    def __init__(self, n, k):
        nn.Module.__init__(self)
        self.n = n
        self.k = k

        self.v = torch.empty(k, n)
        nn.init.xavier_uniform_(self.v)
        self.v = nn.Parameter(self.v)


    def forward(self, x):

        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)

        out = 0.5 * (torch.pow(inter_part1, 2) - inter_part2)

        return out

class CrossNet(nn.Module):
    def __init__(self, n, order=3):
        nn.Module.__init__(self)
        self.n = n
        self.order = order

        self.w = torch.empty(order, n, 1)
        nn.init.xavier_uniform_(self.w)
        self.w = nn.Parameter(self.w)

        self.b = torch.empty(order, n, 1)
        nn.init.xavier_uniform_(self.b)
        self.b = nn.Parameter(self.b)

    def forward(self, x):
        x = x.contiguous().reshape(-1, self.n, 1).contiguous()

        out_list = []
        x_k = x
        for i in range(self.order):
            # print(i)
            # print(x_k.shape)
            temp = torch.matmul(x, x_k.transpose(1, 2).contiguous())
            # print(temp.shape)
            x_k = torch.matmul(temp, self.w[i, :, :]) + self.b[i, :, :] + x_k
            out_list.append(x_k.reshape(-1, self.n).contiguous())

        return out_list

class CompressedInteractionNet(nn.Module):
    def __init__(self, m, H, Hk, vk):
        nn.Module.__init__(self)

        self.Vm = torch.empty(Hk, 1, m, vk)
        nn.init.xavier_uniform_(self.Vm)
        self.Vm = nn.Parameter(self.Vm)

        self.Vh = torch.empty(Hk, 1, vk, H)
        nn.init.xavier_uniform_(self.Vh)
        self.Vh = nn.Parameter(self.Vh)



    def forward(self, x_0, x_h):
        x_0 = x_0.transpose(1, 2).contiguous().reshape(x_0.shape[0], x_0.shape[2], x_0.shape[1], 1)
        x_h = x_0.transpose(1, 2).contiguous().reshape(x_h.shape[0], x_h.shape[2], x_h.shape[1], 1)
        x_h = x_h.transpose(2, 3).contiguous()

        x_k = torch.matmul(x_0, x_h)

        x_k = x_k.reshape(x_k.shape[0], 1, x_k.shape[1], x_k.shape[2], x_k.shape[3]).contiguous()
        self.w = torch.matmul(self.Vm, self.Vh)
        x_k = torch.mul(x_k, self.w)

        x_k = torch.sum(x_k, dim=(3, 4))

        return x_k

class AttentionFFM(nn.Module):
    def __init__(self, m, k):
        nn.Module.__init__(self)

        self.vk = torch.empty(m, k)
        nn.init.xavier_uniform_(self.vk)
        self.vk = nn.Parameter(self.vk)

        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):

        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[2], x.shape[1], 1)
        x_ = x.transpose(2, 3).contiguous()
        x = torch.matmul(x, x_)
        w = torch.matmul(self.vk, self.vk.transpose(0, 1).contiguous())

        x_p = torch.mul(x, w)
        proba = self.softmax(x_p)

        x = torch.sum(torch.mul(x, proba), dim=3)
        x = x.transpose(1, 2).contiguous()
        return x


class myBaseModule():
    def __init__(self, random_seed, input_type=torch.float32, output_type=torch.float32):
        self.random_seed = random_seed
        set_random_seed(self.random_seed)

        self.input_type = input_type
        self.output_type = output_type

    def fit(self, x, y, epoch_nums, batch_size, valid_x, valid_y, custom_metric=None, plot_fold=None):

        if self.vis is not None:
            vis = visdom.Visdom(env=self.vis)

        set_random_seed(self.random_seed)

        self.batch_size = batch_size

        x_train = torch.tensor(x, dtype=self.input_type).cuda()
        y_train = torch.tensor(y, dtype=self.output_type).cuda()
        x_val = torch.tensor(valid_x, dtype=self.input_type).cuda()
        y_val = torch.tensor(valid_y, dtype=self.output_type).cuda()

        loss_fn = self.loss_fn

        optimizer = self.optimizer

        scheduler = self.scheduler

        train = torch.utils.data.TensorDataset(x_train, y_train)
        valid = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        for epoch in range(epoch_nums):
            scheduler.step()
            # print('lr:\t', scheduler.get_lr()[0])

            start_time = time.time()
            self.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):

                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)
            avg_loss /= batch_size

            self.eval()
            valid_preds = np.zeros((valid_y.shape[0]))

            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = self(x_batch)
                y_pred = y_pred.detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)


                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()
            avg_val_loss /= batch_size

            if custom_metric is not None:
                score = custom_metric(valid_y, valid_preds)

            elapsed_time = time.time() - start_time

            if self.vis is not None:
                vis.line(X=torch.Tensor([[epoch, epoch]]),
                         Y=torch.Tensor([[avg_loss, avg_val_loss]]),
                         win='loss'+plot_fold,
                         opts={'legend':['local_loss', 'valid_loss'],
                               'xlabel': 'epoch',
                               'title': 'train'+plot_fold},
                         update='append' if epoch > 0 else None)

            if custom_metric is not None:
                if self.vis is not None:
                    vis.line(X=torch.Tensor([epoch]),
                             Y=torch.Tensor([score]),
                             win='score'+plot_fold,
                             opts={'legend':['score'],
                                   'xlabel': 'epoch',
                                   'title': 'valid'+plot_fold},
                             update='append' if epoch > 0 else None)

            if custom_metric is not None:
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epoch_nums, avg_loss, avg_val_loss, score, elapsed_time))
            else:
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epoch_nums, avg_loss, avg_val_loss, elapsed_time))

    def predict_proba(self, x):
        set_random_seed(self.random_seed)
        x_cuda = torch.tensor(x, dtype=self.input_type).cuda()
        test = torch.utils.data.TensorDataset(x_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

        batch_size = self.batch_size

        self.eval()
        test_preds = np.zeros(len(x))
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = self(x_batch)
            y_pred = y_pred.detach()
            test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()

        return test_preds
    def predict(self, x):
        return self.predict_proba(x)
