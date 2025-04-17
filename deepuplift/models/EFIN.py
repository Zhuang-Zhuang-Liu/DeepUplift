import torch
import torch.nn as nn
from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit, SelfAttentionUnit


class EFIN(BaseModel):
    """
    # input_dim (int): 输入维度（输入特征的维度）
    # hc_dim (int): control net 和uplift net的隐藏层维度
    # hu_dim (int): 交互注意力和表示部分的隐藏单元维度
    # is_self (bool): 是否包含自注意力模块
    # act_type (str): 激活函数类型，默认为'elu'
    """
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, func = nn.ELU() ):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self
        self.act = func

        ''' Feature encoder模块'''
        self.x_rep = nn.Embedding(input_dim, hu_dim) #representation parts for X
        self.t_rep = nn.Linear(1, hu_dim)  # representation parts for T
        
        ''' self-attention模块'''
        self.self_attention = SelfAttentionUnit(hidden_dim=hu_dim ) if is_self else None

        ''' interaction attention模块 '''
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        '''control net'''
        control_hidden_dims = [hc_dim, hc_dim, hc_dim // 2, hc_dim // 4]
        if is_self:
            control_hidden_dims.append(hc_dim // 8)
        self.control_net = TowerUnit( input_dim=input_dim * hu_dim, hidden_dims=control_hidden_dims,
                                      activation=func,task='share',
                                      share_output_dim=hc_dim // 8 if is_self else hc_dim // 4
                                    )
        self.c_logit = nn.Linear(hc_dim // 8 if is_self else hc_dim // 4, 1)
        self.c_tau = nn.Linear(hc_dim // 8 if is_self else hc_dim // 4, 1)


        '''uplift net'''
        uplift_hidden_dims = [hu_dim, hu_dim // 2, hu_dim // 4]
        if is_self:
            uplift_hidden_dims.append(hu_dim // 8)
        self.uplift_net = TowerUnit(input_dim=hu_dim,  hidden_dims=uplift_hidden_dims,
                                    activation=func, task='share',
                                    share_output_dim=hu_dim // 8 if is_self else hu_dim // 4
                                    )
        self.t_logit = nn.Linear(hu_dim // 8 if is_self else hu_dim // 4, 1)
        self.u_tau = nn.Linear(hu_dim // 8 if is_self else hu_dim // 4, 1)


    def interaction_attn(self, t, x):
        attention = []  # 循环计算每个特征对应的注意力中间结果
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

    
    def forward(self, x, t):
        # 输入处理
        t = t.squeeze()
        t_true = torch.unsqueeze(t, 1) #给输入张量增加1个维度
        x_rep = x.unsqueeze(2) * self.x_rep.weight.unsqueeze(0) # 特征编码

        # self-attention
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True) # 归一化处理
        if self.is_self:
            xx, xx_weight = self.self_attention(_x_rep) # 使用新的SelfAttentionUnit
            _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))
        else:
            _x_rep = torch.reshape(_x_rep, (dims[0], dims[1] * dims[2]))

        # control net
        c_last = self.control_net(_x_rep)
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))   # treatment的Feature encoder
        xt, xt_weight = self.interaction_attn(t_rep, x_rep) # xt：交叉注意力后的xt
        u_last = self.uplift_net(xt)
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return  t_logit,[c_logit, u_tau + c_logit],u_tau   # t_pred,y_preds,*eps



def efin_loss(t_logit, y_preds,t, label_list,u_tau):
    t = t.squeeze()
    c_logit = y_preds[0]
    c_logit_fix = c_logit.detach()  # detach() 函数用于从计算图中分离出一个张量，得到一个新的张量，该张量与原始张量共享数据，但不参与梯度计算。
                                     # 如果不这样做，会导致梯度计算时出现问题，因为原始张量和分离后的张量会共享梯度信息，从而导致梯度计算错误。
    uc = c_logit    # y_preds[0]
    ut = (c_logit_fix + u_tau)   #y_preds[1]

    y_true = torch.unsqueeze(label_list, 1)
    t_true = torch.unsqueeze(t, 1)

    # response loss
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    temp = torch.square((1 - t_true) * uc + t_true * ut - y_true) 
    loss1 = torch.mean(temp)
    loss2 = criterion(t_logit, (1 - t_true))   #对t_true取反，进行消偏
    loss = loss1 + loss2
    return loss,loss1,loss2