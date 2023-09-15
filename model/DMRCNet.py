#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x   # pi
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1])) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2])) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x     # e1   B,3,N
    edge2 = neighbor_2nd-org_x     # e2   B,3,N
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N      ni
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N  l1 
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N   l2

    new_pts = torch.cat((org_x,  edge1, edge2 ,dist1, dist2), 1) # B,8,N
    return new_pts




class FAM(nn.Module):
    def __init__(self, channels, ratio = 8):
        super(FAM, self).__init__()

        self.bn1 = nn.BatchNorm1d(channels // ratio)
        self.bn2 = nn.BatchNorm1d(channels // ratio)
        self.bn3 = nn.BatchNorm1d(channels)

        self.q_conv = nn.Conv1d(channels, channels // ratio, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // ratio, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.zeros(1))
    # def forward(self, x):
    #     x_q = self.act(self.bn1(self.q_conv(x))).permute(0, 2, 1)  # b, n, c/ratio
    #     x_k = self.act(self.bn2(self.k_conv(x)))  # b, c/ratio, n
    #     x_v = self.act(self.bn3(self.v_conv(x)))  # b, c, n
    #     energy = torch.bmm(x_q, x_k)  # b, n, n
    #     attention = self.softmax(energy)
    #     attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
    #
    #     x_r = torch.bmm(x_v, attention)  # b, c, n
    #     x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
    #     x = x + x_r
    #     return x

    def forward(self, x):
        _, val_a, val_b = x.size()
        x_q = self.act(self.bn1(self.q_conv(x))).permute(0, 2, 1)  # b, n, c/ratio
        x_k = self.act(self.bn2(self.k_conv(x)))  # b, c/ratio, n
        x_v = self.act(self.bn3(self.v_conv(x)))  # b, c, n
        energy1 = torch.bmm(x_q, x_k)  # b, n, n
        energy2 = torch.matmul(x_q.sum(dim=2),-1*x_q.sum(dim=1))
        energy = energy1+energy2
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


# def forward(self, x):
#     _, val_a, val_b = x.size()
#     x_q = self.act(self.bn1(self.q_conv(x))).permute(0, 2, 1)  # b, n, c/ratio
#     x_k = self.act(self.bn2(self.k_conv(x)))  # b, c/ratio, n
#     x_v = self.act(self.bn3(self.v_conv(x)))  # b, c, n
#     energy1 = torch.bmm(x_q, x_k)  # b, n, n
#     energy2 = torch.matmul(x_q.sum(dim=2), -1 * x_q.sum(dim=1))
#     energy = energy1 + energy2
#     attention = self.softmax(energy)
#     attention = torch.max(attention, -1, keepdim=True)[0].expand_as(attention)-attention
#   #  attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
#
#     x_r = torch.bmm(x_v, attention)  # b, c, n
#     x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
#     x = self.alpha * x_r + x
#  #   x = x + x_r
#     return x

                

class DMRCNet(nn.Module):
    def __init__(self):
        super(GDANET, self).__init__()
        
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv11 = nn.Conv1d(11, 64, kernel_size=1, bias=False)
        self.conv12 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv13 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv14 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn11 = nn.BatchNorm1d(64)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn13 = nn.BatchNorm1d(128)
        self.bn14 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(11*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

        self.alpha = nn.Parameter(torch.zeros(1))
        
        
        
        self.of1 = FAM(64)
        self.of2 = FAM(64)
        self.of3 = FAM(128)
        self.of4 = FAM(256)
        



    def forward(self, x):
        batch_size = x.size(0)

      
        xm = geometric_point_descriptor(x)
     
        x = get_graph_feature(xm, k=20)
        x = self.conv1(x)   # ([B, 64, 1024, 20])
        x = x.max(dim=-1, keepdim=False)[0]
        x1 = self.of1(x)

        x2 = F.relu(self.bn11(self.conv11(xm)))
        # residual connection with a learnable weight
        out1 = self.alpha*x1 + x2
        

        x = get_graph_feature(out1, k=20)     
        x = self.conv2(x)                      
        x = x.max(dim=-1, keepdim=False)[0]
        x1 = self.of2(x)
        
        x2 = F.relu(self.bn12(self.conv12(out1)))
        out2 = self.alpha*x1 +  x2

        x = get_graph_feature(out2, k=20)     
        x = self.conv3(x)                        
        x = x.max(dim=-1, keepdim=False)[0]   

        x1 = self.of3(x) 
        x2 = F.relu(self.bn13(self.conv13(out2)))
        out3 = self.alpha*x1 + x2
        
        x = get_graph_feature(out3, k=20)    
        x = self.conv4(x)                        
        x = x.max(dim=-1, keepdim=False)[0]   

        x1 = self.of4(x) 
        x2 = F.relu(self.bn14(self.conv14(out3)))
        out4 = self.alpha*x1 + x2
        
        x = torch.cat((out1, out2, out3, out4), dim=1) 

        x = self.conv5(x)      # 16,1024,1024
                        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)       # 16,1024    
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)      
        x = torch.cat((x1, x2), 1)     # 16,2048
          

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) 
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) 
        x = self.dp2(x)
        x = self.linear3(x)                                            
        
        return x

    eigenvals, eigenvecs = torch.linalg.eig(mat, eigenvectors=True)


