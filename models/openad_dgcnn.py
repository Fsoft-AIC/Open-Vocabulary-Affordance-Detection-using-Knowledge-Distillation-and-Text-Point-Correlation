import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import clip
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d
import torch_scatter
import math
from .utils.pc_utils import nn_distance, index_point
from torch_cluster import fps
from torch import nn
from einops.layers.torch import Rearrange
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

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
    # print('xuong day')
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
    self.bn5 = nn.BatchNorm2d(512)
    self.bn6 = nn.BatchNorm2d(515)
    self.conv5 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
    self.conv6 = nn.Sequential(nn.Conv2d(2048, 515, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))                              
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
 
  def forward(self, x, y):
    x = get_graph_feature(x, k=40)
    
    x= self.conv5(x)
    x= x.permute(0,2,1,3)
    x = self.conv6(x)
    x= x.permute(0,2,1,3)
    x = x.max(dim=-1, keepdim=True)[0].squeeze(dim=-1)

    y = get_graph_feature(y, k=40)
    
    y= self.conv5(y)
    y= y.permute(0,2,1,3)
    y = self.conv6(y)
    y= y.permute(0,2,1,3)
    y = y.max(dim=-1, keepdim=True)[0].squeeze(dim=-1)

    # print('input affet feature:', x.shape)
    query=self.key_lyr(x)
    # print('query: ',query.shape)
    key=self.query_lyr(y)
    value=self.value_lyr(y)
     
    dotp=torch.matmul(query, key.transpose(-1, -2))*self.scale
    # print(dotp)
     
    scores=self.attention_scores(dotp)
     
    scores=self.dropout(scores)
     
    weighted=torch.matmul(scores, value)
    # print(weighted.shape)
     
    out=self.out_layer(weighted)
     
    return out


# function to find kNN indexes of every points in the cloud
def knn(x, k):      # shape of x: batch_size x fdim x 2048
    inner = -2*torch.matmul(x.transpose(2, 1), x)       # batch_size x 2048 x 2048
    xx = torch.sum(x**2, dim=1, keepdim=True)       # batch_size x 1 x 2048
    pairwise_distance = -xx - inner - xx.transpose(2, 1)        # batch_size x 2048 x 2048

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        # batch_size x 2048 x k
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):       # what is dim9?
    batch_size = x.size(0)      # get the batch_size
    num_points = x.size(2)      # get number of points
    x = x.view(batch_size, -1, num_points)      # maybe not change the shape of x
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)       # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points     # 16 x 1 x 1 [[[0, 2048, 4096, 6144...

    idx = idx.cuda() + idx_base.cuda()

    idx = idx.view(-1)      # 1D vector

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Convolution layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        # Linear layers
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)        # get a 3*3 vector
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv2(x)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = self.conv3(x)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        # (batch_size, 512) -> (batch_size, 256)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        # (batch_size, 256) -> (batch_size, 3*3)
        x = self.transform(x)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        x = x.view(batch_size, 3, 3)

        return x

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

class OpenAD_DGCNN(nn.Module):
    def __init__(self, args, num_classes):
        super(OpenAD_DGCNN, self).__init__()
        self.args = args
        self.k = 40 #args.k
        self.k_relations = 12
        self.k_probs = 12
        self.transform_net = Transform_Net(args)
        self.num_classes = num_classes

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)#args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn_local = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.convlocal = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn_local,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv9 = nn.Conv1d(256, 512, 1)
        self.bn11 = nn.BatchNorm1d(512)
        self.attention = MultiHeadedSelfAttention(515,64,12,0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, affordance):
        batch_size = x.size(0)      # get the batch size
        num_points = x.size(2)      # get the number of points
        point = x.contiguous()
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)     # block matrix multiplication
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]
        # print('check:',x3.shape)
        # (batch_size, 64*3, num_points)
        x = torch.cat((x1, x2, x3), dim=1)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv6(x)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        # (batch_size, num_categoties, 1)
        # l = l.view(batch_size, -1, 1)
        # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        # (batch_size, 1088, num_points)
        x = x.repeat(1, 1, num_points)

        # (batch_size, 1088+64*3, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)
        
        x = self.convlocal(x)
        global_feature = x
        # print('student',x_local.shape)
        
        local_feature = (self.geometry_aware_structure(point[:, :3, :], x[:, :, :])).mean(2)

        # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.conv8(x)
        

        x = self.bn11(self.conv9(x)).permute(0, 2, 1).float()
        with torch.no_grad():
            text_features_org = cls_encoder(affordance)    # il [512, 19]
        # print("text: ",text_features.shape)
        # x = (self.logit_scale * (x @ text_features_org) / (torch.norm(x, dim=2, keepdim=True) @ torch.norm(text_features_org, dim=0, keepdim=True))).permute(0, 2, 1)
        x_final = [] 
        for i in range(x.shape[0]):
            x_batch = x[i].unsqueeze(0)
            text_attention_query_clouds = xattn_score_t2i(x_batch,text_features_org,'softmax')
            # text_features = text_attention_query_clouds.permute(0,2,1).squeeze(0)*text_features_org
            text_features = text_attention_query_clouds.permute(0,2,1).squeeze(0)
            # print('text in point cloud: ',text_attention_query_clouds.shape)
            x_f = (self.logit_scale * (x_batch @ text_features) / (torch.norm(x_batch, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1)
            x_final.append(x_f)
        x = torch.cat(x_final, 0)
        # print('org',x.shape)
        x = F.log_softmax(x, dim=1)
        # local_in_global = self.attention(global_feature)
        local_in_global = global_feature#self.attention(global_feature)
        return x,local_feature,local_in_global
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


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
if __name__ == '__main__':
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
    
    model = OpenAD_DGCNN(None, 19)
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
    model_kd = MultiHeadedSelfAttention(515,64,12,0.1)
    # model_kd = KD_Part(dim_cylinder_list=[64], dim_polar_list=[64])
    model_kd = model_kd.cuda()
    xyz = torch.rand(4, 3, 2048).cuda()

    a,local,blobal =model(xyz,'a')
    print('local',local.shape)
    # print(blobal.shape)
    # blobal = blobal.permute(0,2,1)
    kd_cylinder = model_kd(blobal,blobal)
   
    print(kd_cylinder.shape)
    blobal = blobal.permute(0,2,1)
    # print(kd_polar[0].shape)

    # print(model(xyz,'a').shape)
