import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel

class DragonNet(BaseModel):
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super().__init__() 
        self.fc1 = nn.Linear(in_features=input_dim, out_features=shared_hidden)
        self.fc2 = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)
        self.fcz = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)

        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)

        self.y0_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y0_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y0_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.y1_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y1_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y1_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)
        
    def forward(self, inputs,tr=None):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fcz(x))

        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.relu(self.y0_fc1(z))
        y0 = F.relu(self.y0_fc2(y0))
        y0 = self.y0_out(y0)

        y1 = F.relu(self.y1_fc1(z))
        y1 = F.relu(self.y1_fc2(y1))
        y1 = self.y1_out(y1)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])
        y_preds = [y0, y1]
        return t_pred,y_preds,eps



def dragonnet_loss( t_pred, y_preds,t_true,y_true, eps,alpha=1.0,beta=1.0,tarreg=True):
    """
    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y_preds:[torch.Tensor,torch.Tensor]
        y0_pred: Predicted target variable under control,y1_pred: 
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1 （ 调节不同部分损失在整体损失计算中的相对重要性 ）
    beta: float
        targeted regularization hyperparameter between 0 and 1 （ 用于控制靶向正则化部分在最终损失计算中的影响力 ）
    tarreg: bool
        whether to use targeted regularization
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02  # 归一化处理（ 避免出现0）
    treatment_loss = torch.sum(F.binary_cross_entropy(t_pred, t_true))   # 倾向性loss（交叉熵）

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y_preds[0]))   # 对照组loss（均方误差，MSE Mean Squared Error）
    loss1 = torch.sum(t_true * torch.square(y_true - y_preds[1]))  # 实验组loss（均方误差，MSE Mean Squared Error）
    outcome_loss = loss0 + loss1
    loss = outcome_loss + alpha * treatment_loss

    if tarreg:
        y_pred = t_true * y_preds[1] + (1 - t_true) * y_preds[0]   # 按实际的t预测的y
        # h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))  
        # y_pert = y_pred + eps * h  # 源码：按实际的t预测的y + 动态调整的eps参数 * h 
        y_pert = y_pred + eps # 可渐进一致 ：2个塔的loss一起下降, 去掉eps无法渐进一致
        targeted_regularization = torch.sum((y_true - y_pert)**2)
        loss = loss + beta * targeted_regularization   

    return loss, outcome_loss, treatment_loss




