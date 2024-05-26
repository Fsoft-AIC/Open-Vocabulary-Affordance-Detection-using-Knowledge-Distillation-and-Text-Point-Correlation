import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, MultiStepLR
from dataset import *
from models import *
import loss
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import torch.optim as optim

# Pools of models, optimizers, weights initialization methods, schedulers
model_pool = {
    'openad_pn2': OpenAD_PN2,
    'openad_dgcnn': OpenAD_DGCNN,
    'teacher_point2': teacher_point2,
    'dgcnn_cls_teacher': dgcnn_cls_teacher,
    'dgcnn_partseg_teacher': dgcnn_partseg_teacher,
}

optim_pool = {
    'sgd': SGD,
    'adam': Adam
}

init_pool = {
    'pn2_init': weights_init
}

scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,
    'lr_lambda': LambdaLR,
    'multi_step': MultiStepLR
}


# function to build the model before training
def build_model(cfg, is_teacher=False):
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)     # get the weights initialization method
        model_name = model_info.type        # get the type of network to use (PN2 or DGCNN)
        model_cls = model_pool[model_name] 
        if is_teacher == True:
            num_category = 13
        else:    # get the model name from the pool, model_cls is DGCNN_Estimation or PointNet_Estimation
            num_category = len(cfg.training_cfg.train_affordance)       # get the number of affordance categories
        model = model_cls(model_info, num_category)     # build the model
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)        # apply() function of nn.Module
        return model
    else:
        raise ValueError("Configuration does not have model config!")


# function to build the dataset
def build_dataset(cfg, test=False):
    if hasattr(cfg, 'data'):
        data_info = cfg.data
        data_root = data_info.data_root     # get the path to the dataset
        afford_cat = data_info.category     # get the affordance categories
        if_partial = cfg.training_cfg.get('partial', False)     # partial or not
        if_rotate = cfg.training_cfg.get('rotate', 'None')      # rotate or not
        if_semi = cfg.training_cfg.get('semi', False)       # semi or not
        if_transform = True if if_semi else False       # only for semi task
        # in the Testing mode, not care for now
        if test:
            test_set = AffordNetDataset(
                data_root, 'test', partial=if_partial, rotate=if_rotate, semi=False)
            dataset_dict = dict()
            dataset_dict.update({"test_set": test_set})
            return dataset_dict
        # the training set
        train_set = AffordNetDataset(
            data_root, 'train', partial=if_partial, rotate=if_rotate, semi=if_semi)
        # the validation set
        val_set = AffordNetDataset(
            data_root, 'val', partial=if_partial, rotate=if_rotate, semi=False)
        dataset_dict = dict(
            train_set=train_set,
            val_set=val_set
        )
        # for the semi task, not care for now
        if if_semi:
            train_unlabel_set = AffordNetDataset_Unlabel(data_root)
            dataset_dict.update({"train_unlabel_set": train_unlabel_set})
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


# function to build the loader
def build_loader(cfg, dataset_dict):
    # for the testing set, not care for now
    if "test_set" in dataset_dict:
        test_set = dataset_dict["test_set"]
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        loader_dict = dict()
        loader_dict.update({"test_loader": test_loader})
        return loader_dict
    train_set = dataset_dict["train_set"]
    val_set = dataset_dict["val_set"]
    batch_size_factor = 1 if not cfg.training_cfg.get('semi', False) else 2     # if semi, then load 2
    # training loader
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size // batch_size_factor,
                              shuffle=True, drop_last=True, num_workers=8)
    # validation loader
    val_loader = DataLoader(val_set, batch_size=cfg.training_cfg.batch_size // batch_size_factor,
                            shuffle=False, num_workers=8, drop_last=False)
    loader_dict = dict(
        train_loader=train_loader,
        val_loader=val_loader
    )
    # only for semi task, not care for now
    if "train_unlabel_set" in dataset_dict:
        train_unlabel_set = dataset_dict["train_unlabel_set"]
        train_unlabel_loader = DataLoader(train_unlabel_set, num_workers=4,
                                          batch_size=cfg.training_cfg.batch_size//batch_size_factor, shuffle=True, drop_last=True)
        loader_dict.update({"train_unlabel_loader": train_unlabel_loader})

    return loader_dict


# function to build the loss function
def build_loss(cfg):
    if cfg.training_cfg.get("semi", False):     # for semi task only, not care for now
        loss_fn = loss.VATLoss(warmup_epoch=0)
        # loss_fn = loss.SemiLoss()
    else:
        loss_fn = loss.EstimationLoss()
    return loss_fn


# function to build the optimizer
def build_optimizer(cfg, model, kd_models):
    optim_info = cfg.optimizer
    optim_type = optim_info.type
    optim_info.pop("type")      # get the type of optimizer (adam, sgd,...); then pop it
    optim_cls = optim_pool[optim_type]      # get the name of the optimizer to use
    optimizer = optim_cls([{'params':model.parameters()}, {'params':kd_models.parameters()}], **optim_info)     # construct the optimizer
    scheduler_info = cfg.scheduler
    scheduler_name = scheduler_info.type
    scheduler_info.pop('type')      # get the type of the scheduler (step, cos, lr_lambda,...)
    scheduler_cls = scheduler_pool[scheduler_name]      # get the name of the scheduler to use
    scheduler = scheduler_cls(optimizer, **scheduler_info)      # construct the scheduler
    optim_dict = dict(
        scheduler=scheduler,
        optimizer=optimizer
    )
    return optim_dict
