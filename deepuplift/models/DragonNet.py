import torch
import torch.nn.functional as F

from models.BaseUnit import TowerUnit
from models.BaseModel import BaseModel


class DragonNet(BaseModel):
    def __init__(self, input_dim, share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],base_hidden_dims=[100,100,100,100],
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), 
                 task = 'regression'):
        super().__init__()
        self.share_unit = TowerUnit(input_dim=input_dim, hidden_dims= share_hidden_dims, 
                                    use_batch_norm=True, use_dropout=True, dropout_rate=0.2,
                                    share_output_dim=share_dim, activation= share_hidden_func, task = 'share')
        self.h_1 = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims, 
                            activation= base_hidden_func, task=task,classi_nums=2)
        self.h_0 = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims, 
                            activation= base_hidden_func, task = task,classi_nums=2)
        self.h_t = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims,
                            activation= base_hidden_func,task = 'classification',classi_nums=2)

    def forward(self, x, t):
        phi_x = self.share_unit(x)

        h_1_phi_x = self.h_1(phi_x)  # treat_y
        h_0_phi_x = self.h_0(phi_x)  # control_y
        y_preds = [h_0_phi_x,h_1_phi_x]

        t_pred = self.h_t(phi_x)  # treat_y
        return t_pred,y_preds,phi_x   # pred_y



def dragonnet_loss( t_pred, y_preds,t_true,y_true, eps,alpha=1.0,beta=1.0,tarreg=True, task='regression'):
    """
    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment（softmax输出）
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
    task: str
        'reg' for regression task, 'classi' for classification task
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = t_pred[:, 1]  # 取正类的概率
    t_pred = (t_pred + 0.01) / 1.02  # 归一化处理（ 避免出现0）
    t_pred = t_pred.unsqueeze(1)  # 添加一个维度，使其形状变为 [batch_size, 1]

    treatment_loss = torch.sum(F.binary_cross_entropy(t_pred, t_true))   # 倾向性loss（交叉熵）

    if task == 'regression':
        loss0 = torch.sum((1. - t_true) * torch.square(y_true - y_preds[0]))   # 对照组loss（均方误差，MSE Mean Squared Error）
        loss1 = torch.sum(t_true * torch.square(y_true - y_preds[1]))  # 实验组loss（均方误差，MSE Mean Squared Error）
    elif task == 'classification':
        loss0 = torch.sum((1. - t_true) * F.binary_cross_entropy(y_preds[0], y_true))
        loss1 = torch.sum(t_true * F.binary_cross_entropy(y_preds[1], y_true))
    else:
        raise ValueError("task must be either 'regression' or 'classification'")

    outcome_loss = loss0 + loss1
    loss = outcome_loss + alpha * treatment_loss

    if tarreg:
        y_pred = t_true * y_preds[1] + (1 - t_true) * y_preds[0]   # 按实际的t预测的y
        # h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))  
        # y_pert = y_pred + eps * h  # 源码：按实际的t预测的y + 动态调整的eps参数 * h 

        #y_pert = y_pred + eps # 可渐进一致 ：2个塔的loss一起下降, 去掉eps无法渐进一致
        y_pert = y_pred # ********8

        targeted_regularization = torch.sum((y_true - y_pert)**2)
        loss = loss + beta * targeted_regularization   

    return loss, outcome_loss, treatment_loss