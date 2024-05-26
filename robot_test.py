import os
import torch
import torch.nn as nn
from utils import *
import clip
from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
import torch.nn.functional as F
import numpy as np


# just re-define the point cloud network, do not care
class PN2(nn.Module):
    def __init__(self, normal_channel=False):
        super(PN2, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        # print(l0_points.size())
        x = self.bn1(self.conv1(l0_points))
        return x

if __name__ == "__main__":
    pn2 = PN2().to('cuda')

    checkpoint = './best_model.t7'
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint)

    if exten == '.t7':
        # load checkpoint
        checkpoint_dict = torch.load(checkpoint)
        pn2_dict = pn2.state_dict()
        checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if (k in pn2_dict)}
        pn2_dict.update(checkpoint_dict)
        pn2.load_state_dict(pn2_dict)

    # load clip text encoder
    clip_model, _ = clip.load("ViT-B/32", device='cuda')

    
    xyz = None  # change here, make sure that xyz is a torch.tensor and the shape of xyz is [B, 3, 2048]
    xyz = xyz.float().cuda()
    texts = []  # list of n affordance labels. Remember the background label "none"
    # list of seen affordances is: ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
            #    'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
            #    'listen', 'wear', 'press', 'cut', 'stab', 'none']

    with torch.no_grad():
        pn2.eval()
        clip_model.eval()
        point_features = pn2(xyz).float().permute(0, 2, 1) # [B, 2048, 512]
        tokens = clip.tokenize(texts).to('cuda')

        text_features = clip_model.encode_text(tokens).to('cuda').float().permute(1, 0) # [512, n]
        result = F.softmax(pn2.logit_scale * ((point_features @ text_features) / (torch.norm(point_features, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1), dim=1).cpu().numpy()
        
        result = np.argmax(result, axis=1) # [B, 2048]
        
        # save result
        np.save("result.npy", result)