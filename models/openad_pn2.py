import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

from torch_cluster import fps
from .utils.pc_utils import nn_distance, index_point
from einops.layers.torch import Rearrange
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
# from sklearn.neighbors import NearestNeighbors
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt == "l2norm":
        attn = l2norm(attn, 2)
    elif opt == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def xattn_score_i2t(images, captions, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    # num_points
    for i in range(n_image):
            # Get the i-th text description
            # n_word = cap_lens[i]
            # cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # (n_image, n_word, d)
        # print(captions[i].shape)
        cap_i_expand =   captions.unsqueeze(0).contiguous().permute(0,2,1)               #cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        
        point = images[i].unsqueeze(0)
        print('shape',point.shape)
        weiContext, attn = func_attention(point, cap_i_expand, opt, smooth=9)
        # (n_image, n_region)
        print('text',weiContext.shape)
        row_sim = cosine_similarity(point, weiContext, dim=2)
        print('row',row_sim.shape)
        agg_func = 'LogSumExp'
        if agg_func == 'LogSumExp':
            lambda_lse = 6
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities

def xattn_score_t2i(images, captions, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_image):
        # Get the i-th text description
        # n_word = cap_lens[i]
        # cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = captions.unsqueeze(0).contiguous().permute(0,2,1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        # similarities = []
        points = images[i].unsqueeze(0)
        # for j in range(2048):
        # point = points[j].unsqueeze(0)
        # print('shape',points.shape)
        weiContext, attn = func_attention(cap_i_expand, points, opt, smooth=9)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # print(weiContext.shape)
        # (n_image, n_word)
        # row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        # print('row_sim',row_sim.shape)
        # agg_func = 'LogSumExp'
        # if agg_func == 'LogSumExp':
        #     lambda_lse = 6
        #     row_sim.mul_(lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim)/lambda_lse
        # elif agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        # elif agg_func == 'Mean':
        #     row_sim = row_sim.mean(dim=1, keepdim=True)
        # else:
        #     raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(weiContext)
        # batch_similarities.append(similarities)
    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)
    
    return similarities

class selfAttention(nn.Module):
  def __init__(self, indim, adim, nheads, drop):
    '''
    indim: (int) dimension of input vector
    adim: (int) dimensionality of each attention head
    nheads: (int) number of heads in MHA layer
    drop: (float 0~1) probability of dropping a node
     
    Implements QKV MSA layer
    output = softmax(Q*K/sqrt(d))*V
    scale= 1/sqrt(d), here, d = adim
    '''
    super(selfAttention, self).__init__()
    hdim=adim*nheads
    # print('o day',hdim)
    self.scale= hdim** -0.5 #scale in softmax(Q*K*scale)*V
    self.key_lyr = self.get_qkv_layer(indim, hdim, nheads)
    #nn.Linear(indim, hdim, bias=False) 
    #there should be nheads layers
    self.query_lyr=self.get_qkv_layer(indim, hdim, nheads)
    self.value_lyr=self.get_qkv_layer(indim, hdim, nheads)
     
    self.attention_scores=nn.Softmax(dim=-1)
    self.dropout=nn.Dropout(drop)                      
    self.out_layer=nn.Sequential(Rearrange('bsize nheads indim hdim -> bsize indim (nheads hdim)'),
    nn.Linear(hdim, indim),
    nn.Dropout(drop))
   
  def get_qkv_layer(self, indim, hdim, nheads):
    '''
    returns query, key, value layer (call this function thrice to get all of q, k & v layers)
    '''
    layer=nn.Sequential(nn.Linear(indim, hdim, bias=False),
    Rearrange('bsize indim (nheads hdim) -> bsize nheads indim hdim', nheads=nheads))
     
    return layer
 
  def forward(self, x):
    query=self.query_lyr(x)
    # print('query: ',query.shape)
    key=self.key_lyr(x)
    value=self.value_lyr(x)
     
    dotp=torch.matmul(query, key.transpose(-1, -2))*self.scale
    # print(dotp)
     
    scores=self.attention_scores(dotp)
     
    scores=self.dropout(scores)
     
    weighted=torch.matmul(scores, value)
    # print(weighted.shape)
     
    out=self.out_layer(weighted)
     
    return out
  
class MultiHeadedSelfAttention(nn.Module):
  def __init__(self, indim, adim, nheads, drop):
    '''
    indim: (int) dimension of input vector
    adim: (int) dimensionality of each attention head
    nheads: (int) number of heads in MHA layer
    drop: (float 0~1) probability of dropping a node
     
    Implements QKV MSA layer
    output = softmax(Q*K/sqrt(d))*V
    scale= 1/sqrt(d), here, d = adim
    '''
    super(MultiHeadedSelfAttention, self).__init__()
    hdim=adim*nheads
    # print('o day',hdim)
    self.scale= hdim** -0.5 #scale in softmax(Q*K*scale)*V
    self.key_lyr = self.get_qkv_layer(indim, hdim, nheads)
    #nn.Linear(indim, hdim, bias=False) 
    #there should be nheads layers
    self.query_lyr=self.get_qkv_layer(indim, hdim, nheads)
    self.value_lyr=self.get_qkv_layer(indim, hdim, nheads)
     
    self.attention_scores=nn.Softmax(dim=-1)
    self.dropout=nn.Dropout(drop)
    # self.attention = selfAttention(128,64,12,0.1)                            
    self.out_layer=nn.Sequential(Rearrange('bsize nheads indim hdim -> bsize indim (nheads hdim)'),
    nn.Linear(hdim, indim),
    nn.Dropout(drop))
   
  def get_qkv_layer(self, indim, hdim, nheads):
    '''
    returns query, key, value layer (call this function thrice to get all of q, k & v layers)
    '''
    layer=nn.Sequential(nn.Linear(indim, hdim, bias=False),
    Rearrange('bsize indim (nheads hdim) -> bsize nheads indim hdim', nheads=nheads))
     
    return layer
 
#   def forward(self, x, y):
  def forward(self, x):
    #
    #  x = get_graph_feature(x, k=40)
    
    # x= self.conv5(x)
    # x= x.permute(0,2,1,3)
    # x = self.conv6(x)
    # x= x.permute(0,2,1,3)
    # x = x.max(dim=-1, keepdim=True)[0].squeeze(dim=-1)

    # y = get_graph_feature(y, k=40)
    
    # y= self.conv5(y)
    # x=x.max(dim=-1, keepdim=True)[0].squeeze(dim=-1)
    # y= 
    # y = self.conv6(y)
    # y= y.permute(0,2,1)
    # y = y.max(dim=-1, keepdim=True)[0].squeeze(dim=-1)

    # print('input affet feature:', x.shape)
    x = x.permute(0,2,1)
    # x = self.attention(x)
    
    # y = y.permute(0,2,1)
    # y = self.attention(y)
    query=self.query_lyr(x)
    # print('query: ',query.shape)
    key=self.key_lyr(x)
    value=self.value_lyr(x)
     
    dotp=torch.matmul(query, key.transpose(-1, -2))*self.scale
    # print(dotp)
     
    scores=self.attention_scores(dotp)
     
    scores=self.dropout(scores)
     
    weighted=torch.matmul(scores, value)
    # print(weighted.shape)
     
    out=self.out_layer(weighted)
     
    return out.permute(0,2,1)
class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()
        self.device = torch.device('cuda')
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, classes):
        tokens = clip.tokenize(classes).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).permute(1, 0).float()
        return text_features

cls_encoder = ClassEncoder()

class OpenAD_PN2(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(OpenAD_PN2, self).__init__()
        self.k_relations = 12
        self.k_probs = 12
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
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256,512], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=448, mlp=[128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128])
        
        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        # self.bn_local = nn.BatchNorm1d(1280)
        # self.convlocal = nn.Sequential(nn.Conv1d(128, 1280, kernel_size=1, bias=False),
        #                            self.bn_local,
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, affordance):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        # print(xyz.size())
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            # l0_points = xyz.transpose(1, 2).contiguous()
            # l0_points = None
            l0_xyz = xyz
            l0_points = xyz
        # print('student',l0_xyz.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)     # [B, 128, 2048]
        # print(l0_points.size())
        # l0_points = self.convlocal(l0_points)
        global_feature = l0_points
        # print(global_feature.shape)
        # print('student: ',l0_points_org.shape)
        pc_relations = (self.geometry_aware_structure(xyz[:, :3, :], l0_points[:, :, :])).mean(2)

        l0_points = self.bn1(self.conv1(l0_points))

        # cosine similarity

        l0_points = l0_points.permute(0, 2, 1).float()  # bji, j = 2048, i = 512
        with torch.no_grad():
            text_features_org = cls_encoder(affordance)    # il [512, 19]
        # x = (self.logit_scale * (l0_points @ text_features_org) / (torch.norm(l0_points, dim=2, keepdim=True) @ torch.norm(text_features_org, dim=0, keepdim=True))).permute(0, 2, 1) # blj
        x_final = [] 
        x = l0_points
        for i in range(x.shape[0]):
            x_batch = x[i].unsqueeze(0)
            text_attention_query_clouds = xattn_score_t2i(x_batch,text_features_org,'softmax')
            # text_features = text_attention_query_clouds.permute(0,2,1).squeeze(0)
            text_features = text_attention_query_clouds.permute(0,2,1).squeeze(0) #*text_features_org
            # print('text in point cloud: ',text_attention_query_clouds.shape)
            x_f = (self.logit_scale * (x_batch @ text_features) / (torch.norm(x_batch, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1)
            x_final.append(x_f)
        x = torch.cat(x_final, 0)
        x = F.log_softmax(x, dim=1)
        return x, pc_relations, global_feature

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

class OpenAD_PN2_old(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(OpenAD_PN2_old, self).__init__()
        self.k_relations = 12
        self.k_probs = 12
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

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, affordance):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        # print(xyz.size())
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            # l0_points = xyz.transpose(1, 2).contiguous()
            # l0_points = None
            l0_xyz = xyz
            l0_points = xyz
        # print('student',l0_xyz.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)     # [B, 128, 2048]
        # print(l0_points.size())
        l0_points_org = l0_points

        pc_relations = (self.geometry_aware_structure(xyz[:, :3, :], l0_points_org[:, :, :])).mean(2)

        l0_points = self.bn1(self.conv1(l0_points))

        # cosine similarity

        l0_points = l0_points.permute(0, 2, 1).float()  # bji, j = 2048, i = 512
        with torch.no_grad():
            text_features = cls_encoder(affordance)    # il [512, 19]
        x = (self.logit_scale * (l0_points @ text_features) / (torch.norm(l0_points, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1) # blj
        
        x = F.log_softmax(x, dim=1)
        return x, pc_relations

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

class OpenAD_PN2_large(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(OpenAD_PN2_large, self).__init__()

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

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, affordance):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        # print(xyz.size())
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            # l0_points = xyz.transpose(1, 2).contiguous()
            # l0_points = None
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)     # [B, 128, 2048]
        # print(l0_points.size())

        l0_points = self.bn1(self.conv1(l0_points))

        # cosine similarity

        l0_points = l0_points.permute(0, 2, 1).float()  # bji, j = 2048, i = 512
        with torch.no_grad():
            text_features = cls_encoder(affordance)    # il [512, 19]
        x = (self.logit_scale * (l0_points @ text_features) / (torch.norm(l0_points, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1) # blj
        
        x = F.log_softmax(x, dim=1)
        return x
    

class OpenAD_PN2_LW_no_KD(nn.Module):
    def __init__(self, args, num_classes, normal_channel=False):
        super(OpenAD_PN2_LW_no_KD, self).__init__()
        # self.k_relations = 12
        # self.k_probs = 12
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
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[64, 512], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=448, mlp=[128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128])
        
        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        
                                   

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, affordance):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        # print(xyz.size())
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            # l0_points = xyz.transpose(1, 2).contiguous()
            # l0_points = None
            l0_xyz = xyz
            l0_points = xyz
        # print('student',l0_xyz.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)     # [B, 128, 2048]
        # print(l0_points.size())
        

        

        l0_points = self.bn1(self.conv1(l0_points))

        # cosine similarity

        l0_points = l0_points.permute(0, 2, 1).float()  # bji, j = 2048, i = 512
        with torch.no_grad():
            text_features = cls_encoder(affordance)  # il [512, 19]
        x = (self.logit_scale * (l0_points @ text_features) / (torch.norm(l0_points, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1) # blj
        
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    # import torch
    # import os
    # def get_n_params(model):
    #     pp=0
    #     for p in list(model.parameters()):
    #         nn=1
    #         for s in list(p.size()):
    #             nn = nn*s
    #         pp += nn
    #     return pp
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # model = OpenAD_PN2_large(None, 19)

    # mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    # mem = mem_params + mem_bufs
    
    # print(mem)
    
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

    # print(params)
    # print(get_n_params(model))
    # xyz = torch.rand(16, 3, 2048)

    # # print(model(xyz).size())

    # print(model(xyz,'a').shape)

    import torch
    import os
    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    model = OpenAD_PN2(None, 19)
    model = model.cuda()
    # print(get_n_params(model))

    # mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    # mem = mem_params + mem_bufs
    
    # print(mem)
    
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

    # print(params)

    # print(get_n_params(model))
    model_kd = MultiHeadedSelfAttention(131,64,12,0.1)
    # model_kd = KD_Part(dim_cylinder_list=[64], dim_polar_list=[64])
    model_kd = model_kd.cuda()
    xyz = torch.rand(4, 3, 2048).cuda()

    a,local,blobal =model(xyz,'a')
    print('local',local.shape)
    print('global',blobal.shape)
    # print(blobal.shape)
    # blobal = blobal.permute(0,2,1)
    kd_cylinder = model_kd(local,blobal)
   
    print(kd_cylinder.shape)
    # blobal = blobal.permute(0,2,1)


