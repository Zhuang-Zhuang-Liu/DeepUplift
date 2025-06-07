import torch
import torch.nn as nn
from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit, SelfAttentionUnit


class EFIN(BaseModel):
    """
    # input_dim (int): Input dimension (dimension of input features)
    # hc_dim (int): Hidden layer dimension for control net and uplift net
    # hu_dim (int): Hidden unit dimension for interaction attention and representation parts
    # is_self (bool): Whether to include self-attention module
    # act_type (str): Activation function type, default is 'elu'
    """
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, func = nn.ELU() ):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self
        self.act = func

        ''' Feature encoder module'''
        self.x_rep = nn.Embedding(input_dim, hu_dim) #representation parts for X
        self.t_rep = nn.Linear(1, hu_dim)  # representation parts for T
        
        ''' self-attention module'''
        self.self_attention = SelfAttentionUnit(hidden_dim=hu_dim ) if is_self else None

        ''' interaction attention module '''
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
        attention = []  # Loop to calculate attention intermediate results for each feature
        for i in range(self.nums_feature): 
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        attention = torch.softmax(attention, 1)
        # Add attention dimension
        attention_unsqueezed = torch.unsqueeze(attention, 1)
        matmul_result = torch.matmul(attention_unsqueezed, x)  
        # Remove extra dimension
        outputs = torch.squeeze(matmul_result, 1)
        return outputs, attention

    
    def forward(self, x, t):
        # Input processing
        t = t.squeeze()
        t_true = torch.unsqueeze(t, 1) # Add one dimension to input tensor
        x_rep = x.unsqueeze(2) * self.x_rep.weight.unsqueeze(0) # Feature encoding

        # self-attention
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True) # Normalization processing
        if self.is_self:
            xx, xx_weight = self.self_attention(_x_rep) # Use new SelfAttentionUnit
            _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))
        else:
            _x_rep = torch.reshape(_x_rep, (dims[0], dims[1] * dims[2]))

        # control net
        c_last = self.control_net(_x_rep)
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))   # Feature encoder for treatment
        xt, xt_weight = self.interaction_attn(t_rep, x_rep) # xt: xt after cross attention
        u_last = self.uplift_net(xt)
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return  t_logit,[c_logit, u_tau + c_logit],u_tau   # t_pred,y_preds,*eps



def efin_loss(t_logit, y_preds,t, label_list,u_tau):
    t = t.squeeze()
    c_logit = y_preds[0]
    c_logit_fix = c_logit.detach()  # The detach() function is used to detach a tensor from the computation graph, obtaining a new tensor that shares data with the original tensor but does not participate in gradient computation.
                                     # If this is not done, problems will occur during gradient computation because the original tensor and the detached tensor will share gradient information, leading to incorrect gradient computation.
    uc = c_logit    # y_preds[0]
    ut = (c_logit_fix + u_tau)   #y_preds[1]

    y_true = torch.unsqueeze(label_list, 1)
    t_true = torch.unsqueeze(t, 1)

    # response loss
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    temp = torch.square((1 - t_true) * uc + t_true * ut - y_true) 
    loss1 = torch.mean(temp)
    loss2 = criterion(t_logit, (1 - t_true))   # Invert t_true for debiasing
    loss = loss1 + loss2
    return loss,loss1,loss2