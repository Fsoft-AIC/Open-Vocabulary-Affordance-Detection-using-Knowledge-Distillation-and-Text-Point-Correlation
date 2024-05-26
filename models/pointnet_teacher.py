import torch.nn as nn
import torch.nn.functional as F
from .utild_teacher import PointNetSetAbstraction,PointNetFeaturePropagation
from torch_cluster import fps
from .utils.pc_utils import nn_distance, index_point
from .pointnet_teacher_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch

class teacher_point2(nn.Module):
    def __init__(self,args, num_classes,normal_channel=False):
        super(teacher_point2, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.k_relations = 12
        self.k_probs = 12

    def forward(self, xyz, affordance =''):
        l0_points = torch.cat((xyz,xyz,xyz),axis=1)
        
        l0_xyz = xyz[:,:3,:]
        # print('teacher',l0_xyz.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        l0_points_org = l0_points

        pc_relations = (self.geometry_aware_structure(xyz[:, :3, :], l0_points_org[:, :, :])).mean(2)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, pc_relations, l0_points_org
    def geometry_aware_structure(self, xyz, feat):
        xyz, feat = xyz.transpose(1, 2), feat.transpose(1, 2)
        fps_point_list = []
        fps_feat_list = []
        for batch in range(xyz.shape[0]):
            fps_index = fps(xyz[batch,:, :], None, ratio=0.25, random_start=False).unique()
            fps_point_list.append(xyz[batch, fps_index, :].unsqueeze(0))
            fps_feat_list.append(feat[batch, fps_index, :].unsqueeze(0))

        fps_point = torch.cat(fps_point_list, dim=0)
        fps_feat = torch.cat(fps_feat_list, dim=0)
        pc_dist = nn_distance(fps_point, xyz, pc_distance=True)
        pc_index = (pc_dist.sort(-1, False)[1])[:, :, :self.k_relations]
        index_points_xyz = index_point(xyz, pc_index)
        index_points_features = index_point(feat, pc_index)
        pc_xyz_rel = index_points_xyz - fps_point.unsqueeze(dim=2)
        pc_feat_rel = index_points_features - fps_feat.unsqueeze(dim=2)
        pc_relations = torch.cat([pc_xyz_rel, pc_feat_rel], dim=-1)
        del pc_dist, pc_index, index_points_features
        return pc_relations

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))


# class teacher_point2(nn.Module):
#     def __init__(self,args, num_classes,normal_channel=True):
#         super(teacher_point2, self).__init__()
#         in_channel = 6 if normal_channel else 0
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
#         self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
#         self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, num_classes)
#         self.k_relations = 12
#         self.k_probs = 12

#     def forward(self, xyz,affordance=''):
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         global_feature = l3_points
#         local_feature = (self.geometry_aware_structure(xyz[:, :3, :], global_feature[:, :, :])).mean(2)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)


#         return x,local_feature

#     def geometry_aware_structure(self, xyz, feat):
#         xyz, feat = xyz.transpose(1, 2), feat.transpose(1, 2)
#         fps_point_list = []
#         fps_feat_list = []
#         for batch in range(xyz.shape[0]):
#             fps_index = fps(xyz[batch,:, :], None, ratio=0.25, random_start=False).unique()
#             fps_point_list.append(xyz[batch, fps_index, :].unsqueeze(0))
#             fps_feat_list.append(feat[batch, fps_index, :].unsqueeze(0))

#         fps_point = torch.cat(fps_point_list, dim=0)
#         fps_feat = torch.cat(fps_feat_list, dim=0)
#         pc_dist = nn_distance(fps_point, xyz, pc_distance=True)
#         pc_index = (pc_dist.sort(-1, False)[1])[:, :, :self.k_relations]
#         index_points_xyz = index_point(xyz, pc_index)
#         index_points_features = index_point(feat, pc_index)
#         pc_xyz_rel = index_points_xyz - fps_point.unsqueeze(dim=2)
#         pc_feat_rel = index_points_features - fps_feat.unsqueeze(dim=2)
#         pc_relations = torch.cat([pc_xyz_rel, pc_feat_rel], dim=-1)
#         del pc_dist, pc_index, index_points_features
#         return pc_relations


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss