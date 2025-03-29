import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DragonNet(nn.Module):
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super(DragonNet, self).__init__() # 初始化父类
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

        
    def forward(self, inputs):
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


        t_pred = t_pred
        y_preds = [y0, y1]
        #return y0, y1, t_pred, eps
        return t_pred,y_preds,eps


def dragonnet_loss( t_pred, y_preds,eps,t_true,y_true, alpha=1.0):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------y_preds = [y0_pred, y1_pred]
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02
    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))   # 倾向性loss（交叉熵）

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y_preds[0]))   # 对照组loss（均方误差，MSE Mean Squared Error）
    loss1 = torch.sum(t_true * torch.square(y_true - y_preds[1]))  # 实验组loss（均方误差，MSE Mean Squared Error）
    loss_y = loss0 + loss1

    loss = loss_y + alpha * loss_t

    return loss,loss0,loss1,loss_t   # 全部输出


def tarreg_loss( t_pred, y_preds,eps, t_true,y_true, alpha=1.0, beta=1.0):
    """
    Targeted regularisation（靶向正则化）loss function for dragonnet  # 渐进一致性

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1 （ 调节不同部分损失在整体损失计算中的相对重要性 ）
    beta: float
        targeted regularization hyperparameter between 0 and 1 （ 用于控制靶向正则化部分在最终损失计算中的影响力 ）
    Returns
    loss: torch.Tensor
    """
    vanilla_loss,loss0,loss1,loss_t = dragonnet_loss( t_pred, y_preds, alpha,t_true,y_true)
    
    t_pred = (t_pred + 0.01) / 1.02  # 归一化处理（ 避免出现0）

    y_pred = t_true * y_preds[1] + (1 - t_true) * y_preds[0]   # 按实际的t预测的y

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))  

    #y_pert = y_pred + eps * h  # 源代码：按实际的t预测的y + 动态调整的eps参数 * h ？？
    # y_pert = y_pred   # 会丢失渐进一致性，只有1个塔的loss在下降
    y_pert = y_pred + eps # 可以实现渐进一致 ：2个塔的loss一起下降
    
    targeted_regularization = torch.sum((y_true - y_pert)**2)

    # final
    loss = vanilla_loss + beta * targeted_regularization   
    # loss = vanilla_loss + torch.sum(eps)   # 会丢失渐进一致性，只有1个塔的loss在下降
    outcome_loss = loss0 + loss1
    treatment_loss  = loss_t
    
    return loss, outcome_loss, treatment_loss 


class EarlyStopper:
    # patience（默认值为 15）代表可以容忍验证损失连续不下降的轮次数量，
    # min_delta（默认值为 0）表示验证损失变化的最小差值（只有损失上升幅度超过该差值才会计数）
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


