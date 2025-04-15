import torch
import torch.nn as nn
from models.BaseModel import BaseModel


class EFIN(BaseModel):
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
        self.softmax = nn.Softmax(dim=-1)  
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
        attention = torch.softmax(attention, 1)
        # 增加注意力维度
        attention_unsqueezed = torch.unsqueeze(attention, 1)
        matmul_result = torch.matmul(attention_unsqueezed, x)  
        # 去除多余维度
        outputs = torch.squeeze(matmul_result, 1)
        return outputs, attention

    
    def forward(self, feature_list, is_treat):
        is_treat = is_treat.squeeze()
        t_true = torch.unsqueeze(is_treat, 1) #给输入张量增加1个维度
        x_rep = feature_list.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)

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

        #return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau
        return  t_logit,[c_logit, u_tau + c_logit],u_tau   # t_pred,y_preds,*eps

def efin_loss(t_logit, y_preds,is_treat, label_list,u_tau):
    is_treat = is_treat.squeeze()
    c_logit = y_preds[0]
    c_logit_fix = c_logit.detach()  # detach() 函数用于从计算图中分离出一个张量，得到一个新的张量，该张量与原始张量共享数据，但不参与梯度计算。
                                     # 如果不这样做，会导致梯度计算时出现问题，因为原始张量和分离后的张量会共享梯度信息，从而导致梯度计算错误。
    uc = c_logit    # y_preds[0]
    ut = (c_logit_fix + u_tau)   #y_preds[1]

    y_true = torch.unsqueeze(label_list, 1)
    t_true = torch.unsqueeze(is_treat, 1)

    # response loss
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    temp = torch.square((1 - t_true) * uc + t_true * ut - y_true) 
    loss1 = torch.mean(temp)
    loss2 = criterion(t_logit, (1 - t_true))   #对t_true取反，进行消偏
    loss = loss1 + loss2
    return loss,loss1,loss2



