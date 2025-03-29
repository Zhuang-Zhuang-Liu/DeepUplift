import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, feat_size, embedding_size=4, hidden_dims=[64,32], num_continuous=2):
        super().__init__()
        self.feat_size = feat_size
        self.embedding_size = embedding_size
        self.num_continuous = num_continuous
        self.num_discrete = feat_size - num_continuous
        
        # 离散特征嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(2, embedding_size) for _ in range(self.num_discrete)
        ])
        
        # 连续特征处理层
        self.cont_linear = nn.Linear(num_continuous, embedding_size * num_continuous)
        
        # FM线性部分
        self.fm_linear = nn.Linear(feat_size, 1)
        
        # DNN部分
        dnn_input_dim = (self.num_discrete + num_continuous) * embedding_size
        layers = []
        for dim in hidden_dims:
            layers.extend([nn.Linear(dnn_input_dim, dim), nn.ReLU()])
            dnn_input_dim = dim
        self.dnn = nn.Sequential(*layers)
        self.dnn_fc = nn.Linear(hidden_dims[-1], 1)  # 输出标量用于FM部分
        
        # 最终输出维度（FM标量 + DNN特征）
        self.output_dim = 1 + hidden_dims[-1]

    def forward(self, x):
        # 分离离散和连续特征
        discrete_feats = x[:, :self.num_discrete]  
        cont_feats = x[:, self.num_discrete:]
        
        # 处理离散特征
        if self.num_discrete > 0:
            disc_embeds = [emb(discrete_feats[:,i].long()) for i, emb in enumerate(self.embeddings)]
            disc_embeds = torch.stack(disc_embeds, dim=1)  # [B, num_disc, emb]
        else:
            disc_embeds = torch.empty(x.size(0), 0, self.embedding_size, device=x.device)
        
        # 处理连续特征
        cont_embeds = self.cont_linear(cont_feats)  # [B, num_cont*emb]
        cont_embeds = cont_embeds.view(-1, self.num_continuous, self.embedding_size)  # [B, num_cont, emb]
        
        # 合并所有嵌入
        all_embeds = torch.cat([disc_embeds, cont_embeds], dim=1)  # [B, feat_size, emb]
        
        # FM 的linear part部分-一阶
        fm_first = self.fm_linear(x)  # [B, 1]

        # FM的crossing part部分-二阶
        num_embeds = all_embeds.size(1)
        crossing_terms = []
        for i in range(num_embeds):
            for j in range(i + 1, num_embeds):
                crossing_term = torch.sum(all_embeds[:, i] * all_embeds[:, j], dim=1, keepdim=True)
                crossing_terms.append(crossing_term)
        fm_second = torch.sum(torch.cat(crossing_terms, dim=1), dim=1, keepdim=True)  # [B, 1]
        
        # DNN部分
        dnn_input = all_embeds.view(x.size(0), -1)  # [B, (disc+cont)*emb]
        dnn_out = self.dnn(dnn_input)               # [B, hidden_dim]
        
        # 合并FM和DNN输出
        #print('fm_first',fm_first,'fm_second',fm_second,'dnn_out',dnn_out)
        combined = torch.cat([fm_first + fm_second, dnn_out], dim=1)  # [B, 1 + hidden_dim]
        return combined

class DragonNetDeepFM(nn.Module):
    def __init__(self, input_dim, embedding_size=4,shared_dim=32,num_continuous=2, kind='reg', num_treatments=2):

        super().__init__()
        # 共享层（输出维度由DeepFM保证为1+hidden_dim[-1]）
        self.shared_layer = DeepFM(input_dim,embedding_size,[64, shared_dim-1],num_continuous)

        # 倾向性头*1：用于预测当前样本属于哪个treatment（多分类 ）
        self.treatment_head = nn.Sequential(
            nn.Linear(shared_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_treatments),
            nn.Softmax()  # 兼容单treatment与多treatment
        )

        # outcome头*n：用于预测每个treatment的outcome（回归或二分类）
        self.outcome_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid() if kind == 'classi' else nn.Identity()
            ) for _ in range(num_treatments)
        ])
        

    def forward(self, x):
        shared = self.shared_layer(x)  
        t_pred = self.treatment_head(shared)
        y_preds = [head(shared) for head in self.outcome_heads]
        return t_pred,y_preds


def Dragon_loss(t_pred, y_preds, treatment, outcome, alpha=1.0):
    """
    参数:
        t_pred: [batch_size, num_treatments] 处理预测的softmax输出, 支持单treatment与多treatment
        y_preds: [batch_size, 1] * num_treatments 每个treatment塔的sigmoid输出，支持分类和回归
        treatment: [batch_size] 带有处理索引的张量 (0 到 num_treatments-1)
        outcome: [batch_size] 带有二元标签的张量
        alpha: treatment loss的权重
    返回:
        total_loss: 结果预测损失和处理预测损失的组合
    """
    # 1. 将 y_preds 转换为张量 [batch_size, num_treatments]
    y_pred_matrix = torch.stack(y_preds, dim=1).squeeze(-1)  # [batch_size, num_treatments]
    
    # 2. 结果预测损失: 仅使用实际处理的预测
    treatment = treatment.long()  # 转换为long类型以支持one_hot
    treatment_mask = F.one_hot(treatment, num_classes=len(y_preds)).float().squeeze(1)  # [batch_size, num_treatments]
    #print('treatment_mask',treatment_mask)
    #print('y_pred_matrix',y_pred_matrix)
    selected_y_pred = (y_pred_matrix * treatment_mask).sum(dim=1)  # [batch_size]  只保留实际处理的预测
    #print('selected_y_pred',selected_y_pred,'outcome',outcome)
    outcome_loss = F.binary_cross_entropy(selected_y_pred.unsqueeze(1), outcome.float())
    #print('outcome_loss',outcome_loss)
    
    # 3. 处理预测损失: 交叉熵  
    treatment_loss = F.cross_entropy(t_pred, treatment.squeeze())
    
    # 4. 组合损失
    total_loss = outcome_loss + alpha * treatment_loss
    
    return total_loss, outcome_loss, treatment_loss






if __name__ == "__main__":
    # 生成随机数据
    num_samples = 2
    discrete_features = np.random.randint(0, 2, size=(num_samples, 3))
    continuous_features = np.round(np.random.rand(num_samples, 2), 2)
    sample = np.concatenate([discrete_features, continuous_features], axis=1) # [B, feat_size]
    x = torch.tensor(sample, dtype=torch.float32)  # discrete在前，continuous在后

    # forward
    model = DragonNetDeepFM(input_dim=5, shared_dim=8,num_continuous=2,
                            embedding_size=4, kind='classi', num_treatments=2)
    treatment_preds,outcome_preds = model(x)
    print('treatment_preds: ',treatment_preds)
    print('outcome_preds: ',outcome_preds)
     
    loss  = Dragon_loss(   treatment_preds,
                                    outcome_preds,
                                    treatment=torch.tensor([0,1]),
                                    outcome=torch.tensor([[1.0],[0.0]]), 
                                    alpha=1.0)
    print('loss: ',loss)
    t_pred,y_preds = model(x)
    print('t_pred: ',t_pred, '\ny_preds: ',y_preds)
    
