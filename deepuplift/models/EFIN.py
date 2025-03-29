import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class EFIN(nn.Module):
    """
    EFIN class -- a explicit feature interaction network with two heads.
    # input_dim (int): 输入维度（输入特征的维度）
    # hc_dim (int): control net 和uplift net的隐藏层维度
    # hu_dim (int): 交互注意力和表示部分的隐藏单元维度
    # is_self (bool): 是否包含自注意力模块
    # act_type (str): 激活函数类型，默认为'elu'
    """
    
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, act_type='elu'):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self

        # Feature encoder模块: representation parts for X
        self.x_rep = nn.Embedding(input_dim, hu_dim)
        # Feature encoder模块: representation parts for T
        self.t_rep = nn.Linear(1, hu_dim)
        
        # self-attention模块
        self.softmax = nn.Softmax(dim=-1)  #沿着输入张量的最后1个维度进行计算，
                                        # 比如 (batch_size, sequence_length, num_features) 最后1个维度就是num_features
                                        # 比如 (batch_size, num_features) 最后1个维度就是num_features
        self.Q_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.K_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.V_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)

        # interaction attention模块
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        
        '''control net'''
        self.c_fc1 = nn.Linear(input_dim * hu_dim, hc_dim)
        self.c_fc2 = nn.Linear(hc_dim, hc_dim)
        self.c_fc3 = nn.Linear(hc_dim, hc_dim // 2)
        self.c_fc4 = nn.Linear(hc_dim // 2, hc_dim // 4)
        out_dim = hc_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(hc_dim // 4, hc_dim // 8)
            out_dim = hc_dim // 8

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        '''uplift net'''
        self.u_fc1 = nn.Linear(hu_dim, hu_dim)
        self.u_fc2 = nn.Linear(hu_dim, hu_dim // 2)
        self.u_fc3 = nn.Linear(hu_dim // 2, hu_dim // 4)
        out_dim = hu_dim // 4
        if self.is_self:
            self.u_fc4 = nn.Linear(hu_dim // 4, hu_dim // 8)
            out_dim = hu_dim // 8
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))
            

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        outputs = attn_weights.matmul(V)
        return outputs, attn_weights

    
    def interaction_attn(self, t, x):
        attention = []
        # 循环计算每个特征对应的注意力中间结果
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        # print('interaction attention', attention)
        attention = torch.softmax(attention, 1)
        # print('mean interaction attention', torch.mean(attention, 0))

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    
    def forward(self, feature_list, is_treat):
        t_true = torch.unsqueeze(is_treat, 1) #给输入张量增加1个维度

        x_rep = feature_list.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)
        # *是指逐元素乘法，就是将两个张量对应位置上的元素相乘
        # 将输入的特征list转换为embeding（Feature encoder）
        # 注意这里是用input embeding 与 nn.Embedding(input_dim, hu_dim) 相乘而不是输入到embeding层里  
        # 是 embed 与 矩阵 的常规乘法，而不是按index从embed的对应位置取出确定的向量
        # 也就是说，这里的x不是同质的item，不能用统一的一个矩阵来映射，而是需要矩阵乘法。搞个系数

        # control net
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True) # 归一化处理，torch.linalg.norm ：计算张量范数
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep) # xx: 自注意力交叉后的x

        _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))

        c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(_x_rep))))))))
        # 得到control net的最终结果
        
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
            
        c_logit = self.c_logit(c_last)  # 未经sigmoid激活的net输出结果
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)  # 对logit做sigmoid激活，得到二分类任务的概率

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))   # treatment的Feature encoder
        
        xt, xt_weight = self.interaction_attn(t_rep, x_rep) # xt：交叉注意力后的xt

        u_last = self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(xt))))))
        if self.is_self:
            u_last = self.act(self.u_fc4(u_last))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list, is_treat)
                        # c_tau 无实际意义  # u_tau 代表因果效应

        # regression
        c_logit_fix = c_logit.detach()
        uc = c_logit    # uplift control
        ut = (c_logit_fix + u_tau)   # uplift treatment，其实这里有点像是dipn里的保序层，
                                     # treatment_response  = control * 1 +  uplift * 1 
                                     # control_response  = control * 1 +  uplift * 0 
                                     # 相当于是 直接建模uplift ？

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        # response loss
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        loss1 = torch.mean(temp)
        loss2 = criterion(t_logit, 1 - t_true)   #对t_true取反，进行消偏
        loss = loss1 + loss2

        return loss

## main model
class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, feature_list, is_treat, label_list):
        final_output = self.model.calculate_loss(feature_list, is_treat, label_list)
        return final_output