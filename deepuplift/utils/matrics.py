import torch
from geomloss import SamplesLoss


def wasserstein_torch(X,t ):
    """计算处理组和对照组的Wasserstein距离
    
    Args:
        X: 特征张量
        t: 处理指示变量
        
    Returns:
        float: 两个分布之间的Wasserstein距离
    """
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss

def mmd2_torch(X, t):
    """计算处理组和对照组的MMD距离
    
    Args:
        X: 特征张量
        t: 处理指示变量
        
    Returns:
        float: 两个分布之间的MMD距离
    """
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="energy", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss