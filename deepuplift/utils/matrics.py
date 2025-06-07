import torch
from geomloss import SamplesLoss


def wasserstein_torch(X,t ):
    """Calculate Wasserstein distance between treatment and control groups
    
    Args:
        X: Feature tensor
        t: Treatment indicator variable
        
    Returns:
        float: Wasserstein distance between the two distributions
    """
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss

def mmd2_torch(X, t):
    """Calculate MMD distance between treatment and control groups
    
    Args:
        X: Feature tensor
        t: Treatment indicator variable
        
    Returns:
        float: MMD distance between the two distributions
    """
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="energy", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss