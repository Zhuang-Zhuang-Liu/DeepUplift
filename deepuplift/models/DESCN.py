import torch
import torch.nn as nn
import sys

from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit
from utils.matrics import wasserstein_torch, mmd2_torch


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, do_rate):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu0Network, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu1Network, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim, do_rate):
        super(TauNetwork, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        return tau_logit


class ESX(BaseModel):
    """ESX模型：通过共享网络和多个子网络来预测处理效应
    包含倾向性网络、处理组和对照组网络，以及处理效应网络。
    """
    def __init__(self, input_dim, share_dim, base_dim, 
                do_rate=0.2, use_batch_norm1d=True, normalization="divide", device='cpu'):
        super().__init__()
        
        # 共享网络
        self.share_network = TowerUnit(input_dim=input_dim, 
                                     hidden_dims=[share_dim, share_dim],
                                     share_output_dim=base_dim,
                                     activation=nn.ELU(),
                                     use_batch_norm=use_batch_norm1d,
                                     use_dropout=True,
                                     dropout_rate=do_rate,
                                     task='share',
                                     device=device)
        # 子网络
        self.prpsy_network = PrpsyNetwork(base_dim, do_rate).to(device)
        self.mu1_network = Mu1Network(base_dim, do_rate).to(device)
        self.mu0_network = Mu0Network(base_dim, do_rate).to(device)
        self.tau_network = TauNetwork(base_dim, do_rate).to(device)
        
        self.device = device
        self.to(device)

    def forward(self, x, tr=None):
        """
        Args:
            x: 输入特征
            tr: 处理指示变量（仅用于对齐EFIN接口，无实际用途） 
        Returns:
            tuple: 包含多个预测结果的元组
                - p_prpsy: 倾向性得分
                - [p_h1, p_h0]: 处理组和对照组的预测结果
                - p_estr: 处理组的期望得分
                - p_escr: 对照组的期望得分
                - tau_logit: 处理效应的logit值
                - mu1_logit: 处理组的logit值
                - mu0_logit: 对照组的logit值
                - p_prpsy_logit: 倾向性得分的logit值
                - shared_h: 共享网络的输出
        """
        # 共享网络输出
        shared_h = self.share_network(x)

        # 倾向性得分
        p_prpsy_logit = self.prpsy_network(shared_h)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # 处理组和对照组预测
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)
        p_mu1 = torch.sigmoid(mu1_logit)
        p_mu0 = torch.sigmoid(mu0_logit)
        p_h1 = p_mu1  # 处理组预测
        p_h0 = p_mu0  # 对照组预测

        # 处理效应
        tau_logit = self.tau_network(shared_h)

        # 计算期望得分
        p_estr = torch.mul(p_prpsy, p_h1)  # 处理组期望
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)  # 对照组期望

        return p_prpsy, [p_h1, p_h0], p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy_logit, shared_h



#######################################
def esx_loss(   p_prpsy,y_preds,
                t_labels, y_labels, 
                p_estr, p_escr, tau_logit, mu1_logit, mu0_logit,p_prpsy_logit, shared_h,  # *eps
                # , e_labels,
                prpsy_w=1.0, escvr1_w=1.0, escvr0_w=1.0, h1_w=1.0, h0_w=1.0, 
                mu1hat_w=1.0, mu0hat_w=1.0,imb_dist_w=0.0, imb_dist="wass"):
    """"
    t_pred = p_prpsy_logit
    y_preds = [p_h1, p_h0] 
    """
    p_h1 = y_preds[1]
    p_h0 = y_preds[0]

    p_t = 0.5
    sample_weight = torch.ones_like(t_labels) 
    # define loss function
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    loss_fn = nn.BCELoss()
    loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

    # loss for propensity: 交叉熵
    t_labels = t_labels.squeeze(1)
    prpsy_loss = prpsy_w * loss_w_with_logit_fn(p_prpsy_logit.squeeze(1), t_labels.float()  )     
    
    # loss for ESTR, ESCR
    # p_estr: torch.Size([64, 1]) y_labels: torch.Size([64, 1]) t_labels: torch.Size([64])
    y_labels = y_labels.squeeze(1)
    t_labels = t_labels.float()
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    estr_loss = escvr1_w * loss_w_fn(p_estr, (y_labels * t_labels).float().unsqueeze(1))
    escr_loss = escvr0_w * loss_w_fn(p_escr,(y_labels * (1 - t_labels)).float().unsqueeze(1))

    # loss for TR, CR
    loss_fn = nn.BCELoss()

    tr_loss = h1_w * loss_fn(p_h1,y_labels.unsqueeze(1))  # * (1 / (2 * p_t))
    cr_loss = h0_w * loss_fn(p_h0,y_labels.unsqueeze(1))  # * (1 / (2 * (1 - p_t)))
    # loss for cross TR: mu1_prime, cross CR: mu0_prime
    cross_tr_loss = mu1hat_w * loss_fn(torch.sigmoid(mu0_logit + tau_logit),y_labels.unsqueeze(1))
    cross_cr_loss = mu0hat_w * loss_fn(torch.sigmoid(mu1_logit - tau_logit),y_labels.unsqueeze(1))
    imb_dist_val = 0
    if imb_dist_w > 0:
        if imb_dist == "wass":
            imb_dist_val = wasserstein_torch(X=shared_h, t=t_labels)
        elif imb_dist == "mmd":
            imb_dist_val = mmd2_torch(shared_h, t_labels)
        else:
            sys.exit(1)

    imb_dist_loss = imb_dist_w * imb_dist_val
    loss = prpsy_loss + estr_loss + escr_loss \
                 + tr_loss + cr_loss \
                 + cross_tr_loss + cross_cr_loss \
                 + imb_dist_loss
    outcome_loss = tr_loss + cr_loss
    treatment_loss  = prpsy_loss

    return loss, outcome_loss, treatment_loss
