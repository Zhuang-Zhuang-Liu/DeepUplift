import torch
import torch.nn as nn
from models.BaseModel import BaseModel, BaseLoss
from models.BaseUnit import TowerUnit, SelfAttentionUnit


class EFIN(BaseModel):
    """
    # input_dim (int): Input dimension (dimension of input features)
    # hc_dim (int): Hidden layer dimension for control net and uplift net
    # hu_dim (int): Hidden unit dimension for interaction attention and representation parts
    # is_self (bool): Whether to include self-attention module
    # act_type (str): Activation function type, default is 'elu'
    """
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, task='regression', func=nn.ELU()):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self
        self.task = task
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
        
        '''pred treatment logit'''
        self.t_logit = nn.Linear(hu_dim // 8 if is_self else hu_dim // 4, 1)

        '''pred outcome uplift logit'''
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

        # pred treatment
        t_logit = self.t_logit(u_last) 
        t_pred = torch.sigmoid(t_logit)

        # pred outcome uplift logit
        u_tau = self.u_tau(u_last)

        if self.task == 'classification':
            y0_pred = c_prob
            y1_pred = torch.sigmoid(c_logit.detach() + u_tau) # detach: avoid gradient flow
        elif self.task == 'regression':  
            y0_pred = c_logit
            y1_pred = u_tau + c_logit.detach()  # detach: avoid gradient flow
        else:
            raise ValueError("task must be either 'regression' or 'classification'")
        
        y_preds = [y0_pred, y1_pred]
        return t_pred, y_preds, u_tau



def efin_loss(t_pred, y_preds,t_true, y_true,phi_x=None,task='regression'):
    return  BaseLoss(t_pred=t_pred, y_preds=y_preds, 
                       t_true=t_true, y_true=y_true,
                       loss_type='efin',task=task)

