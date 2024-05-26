# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
import models
import loss
from utils import *
import argparse
# from models.openad_dgcnn import MultiHeadedSelfAttention
from models.openad_pn2 import MultiHeadedSelfAttention

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--config_teacher", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--checkpoint_teacher",
        type=str,
        default=None,
        help="The checkpoint to be resume"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir != None:
        cfg.work_dir = args.work_dir
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    cfg_teacher  = Config.fromfile(args.config_teacher)

    logger = IOStream(opj(cfg.work_dir, 'run.log'))     # opj is os.path.join
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))      # number of GPUs to use
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))      # write to the log file
    if cfg.get('seed') != None:     # set random seed
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    model = build_model(cfg,is_teacher=False).cuda()     # build the model from configuration

    teacher_model = build_model(cfg_teacher, is_teacher=True).cuda()

    # kd_models =  MultiHeadedSelfAttention(515,64,12,0.1)#dgcnn
    kd_models =  MultiHeadedSelfAttention(128,64,12,0.1)#poinet
    kd_models = kd_models.cuda()
    # if train on multiple GPUs, not recommended for PointNet++
    if num_gpu > 1:
        model = nn.DataParallel(model)
        logger.cprint('Use DataParallel!')

    # if the checkpoint file is specified, then load it
    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            teacher_model = nn.DataParallel(teacher_model)
            teacher_model.load_state_dict(torch.load(args.checkpoint_teacher))
            # model = nn.DataParallel(model)
            # model.load_state_dict(torch.load('/home/tuan.vo1/IROS2023_Affordance-master/log/openad_dgcnn/OPENAD_PN2_ESTIMATION_Release_globallocal/best_model.t7'))
        elif exten == '.pth':
            check = torch.load(args.checkpoint_teacher)
            teacher_model.load_state_dict(check['model_state_dict'])
            # check_model = torch.load('/home/tuan.vo1/IROS2023_Affordance-master/log/openad_dgcnn/OPENAD_PN2_ESTIMATION_Release_globallocal/best_model.t7')
            # model.load_state_dict(check_model['model_state_dict'])
    # else train from scratch
    else:
        print("Training from scratch!")

    dataset_dict = build_dataset(cfg)       # build the dataset
    loader_dict = build_loader(cfg, dataset_dict)       # build the loader
    train_loss = build_loss(cfg) 
           # build the loss function
    optim_dict = build_optimizer(cfg, model, kd_models)        # build the optimizer
    # construct the training process
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        loss=train_loss,
        optim_dict=optim_dict,
        logger=logger,
        teacher_model = teacher_model,
        kd_models = kd_models
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()
