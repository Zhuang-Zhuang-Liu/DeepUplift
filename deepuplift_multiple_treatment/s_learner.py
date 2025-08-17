import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Slearner(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),
                 base_hidden_func = torch.nn.ELU(),task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Tarnet',device='cpu'):
        super(Slearner, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)

        # treatment tower
        self.treatment_model = TowerUnit(input_dim = input_dim, 
                hidden_dims=base_hidden_dims, 
                share_output_dim=None, 
                activation=base_hidden_func, 
                use_batch_norm=True, 
                use_dropout=True, 
                dropout_rate=0.3, 
                task=task, 
                classi_nums=classi_nums, 
                output_activation=output_activation_base,
                device=device, 
                use_xavier=True)
    
    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        predcit_pro = self.treatment_model(x).squeeze().unsqueeze(1)

        ate = []

        if not self.training:
            pre = []

            # 复制X_discrete，避免修改原始数据
            X_discrete_modified = X_discrete.clone()

            for treatment_label in self.treatment_label_list:
                # 将最后一列全部赋值为当前treatment_label
                X_discrete_modified[:, -1] = treatment_label

                embedded = [emb(X_discrete_modified[:, i].long()) for i, emb in enumerate(self.embeddings)]
                X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
                x = torch.cat((X_continuous,X_discrete_emb), dim=1)
                pre.append(self.treatment_model(x).squeeze().unsqueeze(1))

            # 计算ATE
            for i in range(1,len(self.treatment_label_list)):
                ate.append(pre[i] - pre[0])

        return torch.cat(ate, dim=1),predcit_pro,None

def slearn_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = y_preds.squeeze().unsqueeze(1)

    if task == 'classification':
        if loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type =='BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("loss_type must be 'BCEWithLogitsLoss' or 'BCELoss'")
    elif task == 'regression':
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type =='huberloss':
            criterion = nn.SmoothL1Loss() 
        else:
            raise ValueError("loss_type must be 'mse' or 'huberloss'")
    else:
        raise ValueError("task must be 'classification' or'regression'")
    
    loss = criterion(y_pred, y_true)
    return loss, 0, None



        
        