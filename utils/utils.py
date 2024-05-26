import numpy as np
import torch
import torch.nn.functional as F
import random


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class PN2_Scheduler(object):
    def __init__(self, init_lr, step, decay_rate, min_lr):
        super().__init__()
        self.init_lr = init_lr
        self.step = step
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        return

    def __call__(self, epoch):
        factor = self.decay_rate**(epoch//self.step)
        if self.init_lr*factor < self.min_lr:
            factor = self.min_lr / self.init_lr
        return factor


class PN2_BNMomentum(object):
    def __init__(self, origin_m, m_decay, step):
        super().__init__()
        self.origin_m = origin_m
        self.m_decay = m_decay
        self.step = step
        return

    def __call__(self, m, epoch):
        momentum = self.origin_m * (self.m_decay**(epoch//self.step))
        if momentum < 0.01:
            momentum = 0.01
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
        return


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import argparse
import socket
import time

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        # self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            # return self.relu(self.bn(self.linear(x)))
            return self.relu(self.linear(x))
        return self.linear(x)
        # return self.bn(self.linear(x))
        

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, n=1):
        super(Embed, self).__init__()

        self.n = n
        if n == 1:
            self.linear = nn.Linear(dim_in, dim_out)
            # self.linear = nn_bn_relu(dim_in, dim_out)
        else:
            # SimCLR claims 2 layer MLP projections work better
            r = 2
            self.l1 = nn_bn_relu(dim_in, dim_in // r)
            self.l2 = nn_bn_relu(dim_in // r, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        if self.n == 1:
            x = self.linear(x)
        else:
            x = self.l1(x, relu=True)
            x = self.l2(x, relu=False)

        return x


class ITLoss(nn.Module):
    """Information-theoretic Loss function
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
    """
    def __init__(self, opt):
        super(ITLoss, self).__init__()
        self.s_dim = 512 #opt.s_dim
        self.t_dim = 512      #opt.t_dim
        self.n_data = 515  #opt.n_data
        self.alpha_it = opt.alpha_it
        self.embed = Embed(512, 512, n=515)

    def forward_correlation_it(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        f_s = self.embed(f_s)

        n, d = f_s.shape

        f_s_norm = (f_s - f_s.mean(0)) / f_s.std(0)
        f_t_norm = (f_t - f_t.mean(0)) / f_t.std(0) 
        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        alpha = self.alpha_it
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(2.0)

        # trace normalisation
        # c_diff = c_diff / torch.sum(c_diff)

        c_diff = c_diff.pow(alpha)

        loss = torch.log2(c_diff.sum())

        return loss

    def forward_mutual_it(self, z_s, z_t):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim, h, w]
            f_t: the feature of teacher network, size [batch_size, t_dim, h, w]

        Returns:
            The IT loss
        """

        f_s = z_s
        f_t = z_t

        if self.s_dim != self.t_dim:
            f_s = self.embed(f_s)

        f_s_norm = F.normalize(f_s)
        f_t_norm = F.normalize(f_t)

        batch_size = f_s.shape[0]
        d = f_s.shape[1]

        # 1. Polynomial kernel
        G_s = torch.einsum('bx,dx->bd', f_s_norm, f_s_norm)
        G_t = torch.einsum('bx,dx->bd', f_t_norm, f_t_norm)
        G_st = G_s * G_t

        # Norm before difference
        z_s = torch.trace(G_s)
        # z_t = torch.trace(G_t)
        z_st = torch.trace(G_st)

        G_s = G_s / z_s
        # G_t = G_t / z_t
        G_st = G_st / z_st

        g_diff = G_s.pow(2) - G_st.pow(2)
        loss = g_diff.sum()

        return loss


# def parse_option():

#     # hostname = socket.gethostname()

#     parser = argparse.ArgumentParser('argument for training')

#     parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
#     parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
#     parser.add_argument('--save_freq', type=int, default=40, help='save frequency')

#     # CIFAR100
#     parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

#     # ImageNet
#     # parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

#     parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
#     parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
#     parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

#     # optimization
#     parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
#     parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
#     parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
#     parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

#     # dataset
#     parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

#     # model
#     parser.add_argument('--model_s', type=str, default='wrn_16_2',
#                         choices=['resnet34', 'resnet18', # ImageNet
#                                  'resnet20_1w1a', # binary
#                                  'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
#                                  'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
#                                  'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
#                                  'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])

#     parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')
#     parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

#     # distillation
#     parser.add_argument('--trial', type=str, default='1', help='trial id')

#     parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
#     parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight balance for other losses')

#     parser.add_argument('--kd_T', type=float, default=4.0, help='temperature for KD distillation')

#     # IT distillation
#     parser.add_argument('--lambda_corr', type=float, default=2.0, help='correlation loss weight')
#     parser.add_argument('--lambda_mutual', type=float, default=1.0, help='mutual information loss weight')
#     parser.add_argument('--alpha_it', type=float, default=1.01, help='Renyis alpha')

#     opt = parser.parse_args()

#     # set different learning rate from these 4 models
#     if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
#         opt.learning_rate = 0.01

#     # set the path according to the environment
#     opt.model_path = './save/student_model'
#     opt.tb_path = './save/student_tensorboards'

#     iterations = opt.lr_decay_epochs.split(',')
#     opt.lr_decay_epochs = list([])
#     for it in iterations:
#         opt.lr_decay_epochs.append(int(it))

#     # opt.model_t = get_teacher_name(opt.path_t)

#     # opt.model_name = 'S:{}_T:{}_{}_r:{}_b:{}_{}_l_corr:{}_l_mutual:{}_alpha:{}'.format(opt.model_s, opt.model_t, opt.dataset,
#     #                                                                                       opt.gamma, opt.beta, opt.trial,
#     #                                                                                       opt.lambda_corr, opt.lambda_mutual, opt.alpha_it)
#     # : is not allowed in Windows paths
#     # opt.model_name = opt.model_name.replace(":", "-")

#     # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
#     # if not os.path.isdir(opt.tb_folder):
#     #     os.makedirs(opt.tb_folder)

#     # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
#     # if not os.path.isdir(opt.save_folder):
#     #     os.makedirs(opt.save_folder)

#     opt.epochs = opt.epochs
#     opt.lr_decay_epochs = [int(epoch) for epoch in opt.lr_decay_epochs]
#     opt.batch_size = int(opt.batch_size)

#     return opt
# opt = parse_option()
# criterion_kd = ITLoss(opt)


