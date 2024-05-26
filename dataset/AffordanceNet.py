# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import h5py
import json
from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y
import pickle as pkl


# just normalize the point cloud
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


# for semi task only, not care for now
def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, rotate='None', semi=False):
        super().__init__()
        self.data_dir = data_dir        # path to the dataset
        self.split = split      # train or val or test

        self.partial = partial      # partial or not
        self.rotate = rotate        # rotate or not
        self.semi = semi        # semi or not

        self.load_data()

        self.affordance = self.all_data[0]["affordance"]

        return

    def load_data(self):
        self.all_data = []
        # only for the semi task, not care for now
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            # if partial
            if self.partial:
                with open(opj(self.data_dir, 'partial_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)     # get the dataset for the partial task according to the specific split
            # if rotate
            elif self.rotate != "None" and self.split != 'train':
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)      # get the dataset for the rotate task according to the specific split
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)     # get the full-shape dataset
            # if just full-shape
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
        for index, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():        # multiple views
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            elif self.split != 'train' and self.rotate != 'None':       # get the point clouds from full-shape dataset
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]        # ID of the shape
                temp_info["semantic class"] = info["semantic class"]        # object category (door, cup,...)
                temp_info["affordance"] = info["affordance"]        # list of 18 affordance categories
                temp_info["data_info"] = info["full_shape"]     # point coordinates and scores for affordance categories
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]        # get the index
        modelid = data_dict["shape_id"]     # get the ID of the object
        modelcat = data_dict["semantic class"]      # get the semantic class (door, cup,...)

        data_info = data_dict["data_info"]      # data information: coordinates and affordance scores
        model_data = data_info["coordinate"].astype(np.float32)     # get the coordinate of point cloud, shape (2048, 3)
        labels = data_info["label"]     # get the affordance scores dictionary
        
        temp = labels.astype(np.float32).reshape(-1, 1)        # shape (2048, 1)
        model_data = np.concatenate((model_data, temp), axis=1)     # when this loop finishes, the shape of model data is (2048, 21)

        datas = model_data[:, :3]       # extract 3 first columns to get the point cloud coordinate
        targets = model_data[:, 3:]     # extract the remain columns to get the affordance scores

        if self.rotate != 'None':
            if self.split == 'train':
                if self.rotate == 'so3':        # if train-rotate-so3
                    datas = rotate_point_cloud_SO3(
                        datas[np.newaxis, :, :]).squeeze()
                elif self.rotate == 'z':        # if train-rotate-z
                    datas = rotate_point_cloud_y(
                        datas[np.newaxis, :, :]).squeeze()
            else:
                r_matrix = data_dict["rotate_matrix"]
                datas = (np.matmul(r_matrix, datas.T)).T

        datas, _, _ = pc_normalize(datas)       # normalize the point cloud

        return datas, datas, targets, modelid, modelcat

    def __len__(self):
        return len(self.all_data)


# only for the semi task, not care for now
class AffordNetDataset_Unlabel(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.load_data()
        self.affordance = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []
        with open(opj(self.data_dir, 'semi_unlabel_1.pkl'), 'rb') as f:
            temp_data = pkl.load(f)
        for info in temp_data:
            temp_info = {}
            temp_info["shape_id"] = info["shape_id"]
            temp_info["semantic class"] = info["semantic class"]
            temp_info["affordance"] = info["affordance"]
            temp_info["data_info"] = info["full_shape"]
            self.all_data.append(temp_info)

    def __getitem__(self, index):
        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        datas = data_info["coordinate"].astype(np.float32)

        datas, _, _ = pc_normalize(datas)

        return datas, datas, modelid, modelcat

    def __len__(self):
        return len(self.all_data)