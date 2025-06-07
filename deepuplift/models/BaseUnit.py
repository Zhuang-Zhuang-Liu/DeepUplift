import torch.nn as nn
import torch

class TowerUnit(nn.Module):
    """
    Classification task: output softmax
    Regression task: output values
    Share: output high-dimensional features
    """
    def __init__(self, input_dim, hidden_dims=[], 
                 share_output_dim=16, activation=nn.ELU(), 
                 use_batch_norm=False, use_dropout=False, dropout_rate=0.2, 
                 task='share', classi_nums=None, 
                 device='cpu', use_xavier=True):
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
            output_dim , output_activation = 1 , None  # No activation function for regression tasks
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
    Self-Attention unit for implementing self-attention mechanism
    Parameters:
         hidden_dim (int): Hidden layer dimension
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.Q_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.K_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.V_w = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Forward propagation
        Parameters:
            x (torch.Tensor): Input tensor with shape [batch_size, seq_len, hidden_dim]
        Returns:
            torch.Tensor: Output tensor with shape [batch_size, seq_len, hidden_dim]
            torch.Tensor: Attention weights with shape [batch_size, seq_len, seq_len]
        """
        Q = self.Q_w(x)
        K = self.K_w(x)
        V = self.V_w(x)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5) # Calculate attention scores
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        outputs = attn_weights.matmul(V) # Apply attention weights
        
        return outputs, attn_weights