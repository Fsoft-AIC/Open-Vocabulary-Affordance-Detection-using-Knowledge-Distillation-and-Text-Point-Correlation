# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from utils.eval import evaluation
from time import time
import torch.optim as optim
from torch.autograd import Variable


import torch as T

# -----------------------------------------------------------

class ContrastiveLoss(T.nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = T.nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      return T.mean(T.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))

class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.writer = SummaryWriter(self.work_dir)
        self.logger = running['logger']
        self.model = running["model"]
        self.teacher_model = running["teacher_model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.val_loader = self.loader_dict.get("val_loader", None)
        self.train_unlabel_loader = self.loader_dict.get(
            "train_unlabel_loader", None)
        if self.train_unlabel_loader is not None:       # for semi task only, not care for now
            self.unlabel_loader_iter = iter(self.train_unlabel_loader)
        self.test_loader = self.loader_dict.get("test_loader", None)
        self.loss = running["loss"]
        self.optim_dict = running["optim_dict"]
        self.optimizer = self.optim_dict.get("optimizer", None)
        self.scheduler = self.optim_dict.get("scheduler", None)
        self.epoch = 0
        self.best_val_mIoU = 0.0
        self.best_val_mIoU_zeroshot = 0.0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
        self.train_affordance = cfg.training_cfg.train_affordance
        self.val_affordance = cfg.training_cfg.val_affordance
        self.zero_shot_Affordance = cfg.training_cfg.zero_affordance
        self.kd_models = running['kd_models']
        self.kd_optimizer  = optim.Adam([{'params':self.kd_models.parameters()}], lr=0.0005)
        self.kd_loss = torch.nn.MSELoss(reduction='mean')
        self.contrastiveloss = ContrastiveLoss()
        
        return

    def train(self):
        train_loss = 0.0
        count = 0.0
        self.model.train()
        self.kd_models.train()
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        for data, data1, label, _, cats in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):
            if self.train_unlabel_loader is not None:
                try:
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()
                except StopIteration:
                    self.unlabel_loader_iter = iter(self.train_unlabel_loader)
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()

            data, label = data.float().cuda(), label.float().cuda()
            
            data = data.permute(0, 2, 1)
            label = torch.squeeze(label).long().contiguous()
            batch_size = data.size()[0]
            num_point = data.size()[2]
            # print('check input:',data.shape)
            self.optimizer.zero_grad()
            if self.train_unlabel_loader is not None:
                ul_data = ul_data.cuda().float().permute(0, 2, 1)
                data_ = torch.cat((data, ul_data), dim=0)  # VAT
                afford_pred = self.model(data_, self.train_affordance)
            else:
                data_st = data[:, :3, :]
                afford_pred, pc_relations_st, local_in_global_st = self.model(data_st, self.train_affordance)
                _, pc_relations_tc, local_in_global_tc = self.teacher_model(data, self.train_affordance)
                # pc_relations_tc = Variable(pc_relations_tc, requires_grad=False)
                # local_in_global_tc = Variable(local_in_global_tc, requires_grad=False)

            afford_pred = afford_pred.contiguous()
            if self.train_unlabel_loader is not None:
                l_pred = afford_pred[:batch_size, :, :]  # VAT
                ul_pred = afford_pred[batch_size:, :, :]  # VAT
                loss = self.loss(self.model, data, ul_data,
                                 l_pred, label, ul_pred, self.epoch)  # VAT
            else:
                loss = self.loss(afford_pred, label)
                # print('vo day')
                # loss_struct_distill =  torch.square(pc_relations_st - pc_relations_tc).mean([2, 1, 0]) #self.kd_loss(pc_relations_st,pc_relations_tc)
                # loss_struct_distill =  self.kd_loss(pc_relations_st,pc_relations_tc)#mse
                # loss_struct_distill = (1 - torch.nn.CosineSimilarity()(pc_relations_st, pc_relations_tc)).mean() #cosine
                loss_struct_distill = self.contrastiveloss(pc_relations_st, pc_relations_tc,0)
                
                # loss_struct_distill_global =   self.kd_loss(l0_points_org_st,l0_points_org_tc) #torch.square(l0_points_org_st - l0_points_org_tc).mean([2, 1, 0])
                # loss_total = loss+0.5*loss_struct_distill+loss_struct_distill_global
                # local_in_global_att = self.kd_models(local_in_global_st,local_in_global_tc) # for dgcnn
                local_in_global_att = self.kd_models(local_in_global_tc) #for_poinet
                # print(local_in_global_att.shape)
                # glob_ftc = self.kd_models(l0_points_org_tc)
                # loss_kd =    torch.square(local_in_global_st-pc_relations_tc).mean([2, 1, 0]) #DGCNN#self.kd_loss(glob_fst,glob_ftc)
                # loss_kd =   torch.square(local_in_global_st-local_in_global_tc).mean([2, 1, 0]) #pointnet
                #loss_kd_gl =   self.kd_loss(local_in_global_st,local_in_global_tc)
                # local_in_global_att = local_in_global_att.permute(0,2,1)
                # print(local_in_global_att.shape)
                # print(pc_relations_st.shape)
                loss_global_local = self.kd_loss(local_in_global_st,local_in_global_att) #mse
                # loss_global_local = (1 - torch.nn.CosineSimilarity()(local_in_global_st, local_in_global_att)).mean()#cosine
                loss_total = loss+loss_struct_distill+  loss_global_local  #+ loss_kd_gl 
                # loss_total = loss
            loss_total.backward()
            # loss_kd.backward(retain_graph=True)
            self.optimizer.step()
            # self.kd_optimizer.step()

            count += batch_size * num_point
            train_loss += loss.item()
        self.scheduler.step()
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        epoch_time = time() - start
        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, train_loss*1.0/num_batches, epoch_time//1)
        self.writer.add_scalar('Loss', train_loss*1.0/num_batches, self.epoch)
        self.logger.cprint(outstr)
        self.epoch += 1

    def val(self):
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch-1))
        mIoU = evaluation(self.logger, self.cfg, self.model,
                         self.val_loader, self.val_affordance)

        mIoU_Zeroshot = evaluation(self.logger, self.cfg, self.model,
                         self.val_loader, self.zero_shot_Affordance)

        if mIoU >= self.best_val_mIoU and mIoU_Zeroshot >= self.best_val_mIoU_zeroshot:
            self.best_val_mIoU = mIoU
            self.best_val_mIoU_zeroshot = mIoU_Zeroshot
            self.logger.cprint('Saving model......')
            self.logger.cprint('Best mIoU: %f' % self.best_val_mIoU)
            self.logger.cprint('Best mIoU_Zero-shot: %f' % self.best_val_mIoU_zeroshot)
            torch.save(self.model.state_dict(),
                   opj(self.work_dir, 'best_model.t7'))

        torch.save(self.model.state_dict(),
                      opj(self.work_dir, 'current_model.t7'))

    def test(self):
        self.logger.cprint('Begin testing......')
        evaluation(self.logger, self.cfg, self.model,
                   self.test_loader, self.val_affordance)
        return

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        if self.test_loader != None:
            epoch_runner = getattr(self, 'test')
            epoch_runner()
        else:
            while self.epoch < EPOCH:
                for key, running_epoch in workflow.items():
                    epoch_runner = getattr(self, key)
                    for e in range(running_epoch):
                        epoch_runner()
