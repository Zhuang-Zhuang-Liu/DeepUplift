import torch.nn as nn

class TowerUnit(nn.Module):
    """"
    分类任务：输出softmax
    回归任务：输出值
    share：输出高维
    """
    def __init__(self, input_dim, hidden_dims=[], 
                 share_output_dim=16, activation=nn.ELU(), 
                 use_batch_norm=False, use_dropout=False, dropout_rate=0.5, 
                 task='share', classi_nums=None):
        super().__init__()
        layers = []

        # hidden layers
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            if use_batch_norm:  layers.append(nn.BatchNorm1d(dim))
            if use_dropout: layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # output layers
        if task == 'classification' :
            if classi_nums is None:
                raise ValueError("classi_nums must be specified for classification task")
            output_dim , output_activation = classi_nums , nn.Softmax(dim=-1)
        elif task == 'regression':
            output_dim , output_activation = 1 , activation
        elif task == 'share':
            output_dim , output_activation = share_output_dim , activation
        else:
            raise ValueError("task must be 'regression', 'classification' or 'share'")
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(output_activation)
        if use_batch_norm:  layers.append(nn.BatchNorm1d(dim))
        if use_dropout: layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)