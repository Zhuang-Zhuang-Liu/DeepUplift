import torch
import torch.nn as nn
import torch.nn.functional as F


class EUEN(nn.Module):
    def __init__(self, input_dim: int, hc_dim: int, hu_dim: int, is_self: bool = False, l2_reg: float = 0.0, act_type: str = "elu"):
        super(EUEN, self).__init__()
        self.hc_dim = hc_dim
        self.hu_dim = hu_dim
        self.is_self = is_self
        self.l2_reg = l2_reg
        self.act_type = act_type

        # 激活函数映射
        self.activation_mapping = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus()
        }

        # 构建控制网络和提升网络
        self.control_net = self._build_network(input_dim, hc_dim, is_self)
        self.uplift_net = self._build_network(input_dim, hu_dim, is_self)

        # 输出层
        self.c_logit = nn.Linear(hc_dim // 8 if is_self else hc_dim // 4, 1)
        self.c_tau = nn.Linear(hc_dim // 8 if is_self else hc_dim // 4, 1)
        self.t_logit = nn.Linear(hu_dim // 8 if is_self else hu_dim // 4, 1)
        self.u_tau = nn.Linear(hu_dim // 8 if is_self else hu_dim // 4, 1)

    def _build_network(self, input_dim: int, hidden_dim: int, is_self: bool) -> nn.Sequential:
        layers = []
        dims = [input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4]
        if is_self:
            dims.append(hidden_dim // 8)

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                self._get_activation()
            ])
        return nn.Sequential(*layers)

    def _get_activation(self):
        return self.activation_mapping.get(self.act_type, nn.Identity())

    def forward(self, x: torch.Tensor, tr: torch.Tensor):
        # 控制网络
        c_features = self.control_net(x)
        c_logit = self.c_logit(c_features)
        c_tau = self.c_tau(c_features)
        c_prob = torch.sigmoid(c_logit)

        # 提升网络
        u_features = self.uplift_net(x)
        t_logit = self.t_logit(u_features)
        u_tau = self.u_tau(u_features)
        t_prob = torch.sigmoid(t_logit)

        # 回归
        c_logit_fix = c_logit.detach()  # 分离 c_logit_fix 以停止梯度计算
        uc = c_logit
        ut = c_logit_fix + u_tau  # 回归任务
        t_pred = None
        return t_pred, [uc, ut]
    
    


def euen_loss(t_pred, y_preds, is_treat, label):
    """
    Calculate the uplift MSE loss
    
    Args:
        t_pred:占位
        label: Tensor of shape [batch_size, 1] containing the true labels
        is_treat: Tensor of shape [batch_size, 1] indicating treatment (1) or control (0)
        concat_pred: Tensor of shape [batch_size, 2] containing predictions for control and treatment
    """
    c_pred = y_preds[0]#.unsqueeze(1)
    t_pred = y_preds[1]#.unsqueeze(1)
    
    is_t = is_treat
    is_c = 1. - is_treat
    
    loss = torch.square(is_c * c_pred + is_t * t_pred - label)
    reduce_loss = torch.mean(loss)
            
    # Calculate MSE
    u = (1 - is_treat) * c_pred + is_treat * t_pred
    outcome_loss = F.mse_loss(u, label)
    treatment_loss = 0 #占位
    
    return reduce_loss, outcome_loss, treatment_loss