import torch.nn as nn
import torch

class TowerUnit(nn.Module):
    """"
    分类任务：输出softmax
    回归任务：输出值
    share：输出高维
    """
    def __init__(self, input_dim, hidden_dims=[], 
                 share_output_dim=16, activation=nn.ELU(), 
                 use_batch_norm=False, use_dropout=False, dropout_rate=0.5, 
                 task='share', classi_nums=None, device='cpu', use_xavier=True):
        super().__init__()
        self.device = device
        layers = []

        # hidden layers
        prev_dim = input_dim
        for dim in hidden_dims:
            linear_layer = nn.Linear(prev_dim, dim)
            if use_xavier:
                nn.init.xavier_uniform_(linear_layer.weight)
                if linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            if use_batch_norm:  
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation)
            if use_dropout: 
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # output layers
        if task == 'classification' :
            if classi_nums is None:
                raise ValueError("classi_nums must be specified for classification task")
            output_dim , output_activation = classi_nums , nn.Softmax(dim=-1)
        elif task == 'regression':
            output_dim , output_activation = 1 , None  # 回归任务不使用激活函数
        elif task == 'share':
            output_dim , output_activation = share_output_dim , activation
        else:
            raise ValueError("task must be 'regression', 'classification' or 'share'")
            
        output_layer = nn.Linear(prev_dim, output_dim)
        if use_xavier:
            nn.init.xavier_uniform_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        if use_batch_norm:  
            layers.append(nn.BatchNorm1d(output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)


class SelfAttentionUnit(nn.Module):
    """
        Self-Attention单元，用于实现自注意力机制
    参数:
         hidden_dim (int): 隐藏层维度
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.Q_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.K_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.V_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, seq_len, hidden_dim]
            torch.Tensor: 注意力权重，形状为 [batch_size, seq_len, seq_len]
        """
        Q = self.Q_w(x)
        K = self.K_w(x)
        V = self.V_w(x)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5) # 计算注意力分数
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        outputs = attn_weights.matmul(V) # 应用注意力权重
        
        return outputs, attn_weights
