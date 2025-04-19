import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class EarlyStopper:
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



class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_dataloader = None
        self.valid_dataloader = None


    def create_dataloaders(self, x, y, t, valid_perc=None,batch_size=64,num_workers=4):
        if valid_perc:
            x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
                x.to_numpy(), y.to_numpy(), t.to_numpy(), 
                test_size=valid_perc, random_state=42
            )
            x_train = torch.Tensor(x_train)
            x_test = torch.Tensor(x_test)
            y_train = torch.Tensor(y_train).reshape(-1, 1)
            y_test = torch.Tensor(y_test).reshape(-1, 1)
            t_train = torch.Tensor(t_train).reshape(-1, 1)
            t_test = torch.Tensor(t_test).reshape(-1, 1)
            train_dataset = TensorDataset(x_train, t_train, y_train)
            valid_dataset = TensorDataset(x_test, t_test, y_test)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
        else:
            x = torch.Tensor(x)
            t = torch.Tensor(t).reshape(-1, 1)
            y = torch.Tensor(y).reshape(-1, 1)
            train_dataset = TensorDataset(x, t, y)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )


    def fit(self, x, y, t,batch_size=64, epochs=10,
            learning_rate=1e-5,valid_perc=None,
            loss_f=None, tensorboard=False
            ):

        model = self.train()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.create_dataloaders(x, y, t, valid_perc,batch_size)
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            import time
            model_name = self.__class__.__name__
            timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
            log_dir = f"runs/{model_name}_{timestamp}"
            writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(epochs):
            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                t_pred,y_preds,*eps = model(X,tr)
                loss, outcome_loss, treatment_loss = loss_f(t_pred, y_preds,tr, y1, *eps)
                optim.zero_grad()
                loss.backward()
                optim.step()

            if self.valid_dataloader:
                model.eval()
                valid_loss,valid_outcome_loss,valid_treatment_loss = self.validate_step(loss_f)
                print(f"""--epoch: {epoch} 
                train_loss: {loss:.4f} outcome_loss:{outcome_loss:.4f} treatment_loss:{treatment_loss:.4f} 
                valid_loss: {valid_loss:.4f} valid_outcome_loss:{valid_outcome_loss:.4f} valid_treatment_loss:{valid_treatment_loss:.4f} """)
                if tensorboard:
                    writer.add_scalar('Loss/train', loss, epoch)
                    writer.add_scalar('Loss/valid', valid_loss, epoch)
                    writer.add_scalar('OutcomeLoss/train', outcome_loss, epoch)
                    writer.add_scalar('OutcomeLoss/valid', valid_outcome_loss, epoch)
                    writer.add_scalar('TreatmentLoss/train', treatment_loss, epoch)
                    writer.add_scalar('TreatmentLoss/valid', valid_treatment_loss, epoch)
                model.train()
                if early_stopper.early_stop(valid_loss):
                    if tensorboard:
                        writer.close()
                    break
            else:
                print(f"""epoch: {epoch} 
                train_loss: {loss:.4f} outcome_loss:{outcome_loss:.4f} treatment_loss:{treatment_loss:.4f} """)
                if tensorboard:
                    writer.add_scalar('Loss/train', loss, epoch)
                    writer.add_scalar('OutcomeLoss/train', outcome_loss, epoch)
                    writer.add_scalar('TreatmentLoss/train', treatment_loss, epoch)


    def validate_step(self,loss_f):
        valid_loss = []
        valid_outcome_loss = []
        valid_treatment_loss = []
        with torch.no_grad(): 
            for batch, (X, tr, y1) in enumerate(self.valid_dataloader):
                t_pred,y_preds, *eps = self.predict(X,tr)
                loss, outcome_loss, treatment_loss = loss_f(t_pred, y_preds,tr,y1, *eps)
                valid_loss.append(loss)
                valid_outcome_loss.append(outcome_loss)        
                valid_treatment_loss.append(treatment_loss)
        return torch.Tensor(valid_loss).mean(),  \
               torch.Tensor(valid_outcome_loss).mean(), \
               torch.Tensor(valid_treatment_loss).mean()


    def predict(self, x,tr=None):
        model = self.train()
        if isinstance(x, pd.DataFrame):  # 推理阶段: 如果x是df就转为numpy
            x = x.to_numpy()
        if tr is not None:
            if isinstance(tr, pd.Series):
                tr = tr.to_numpy()
            tr = torch.Tensor(tr)
        x = torch.Tensor(x)
        with torch.no_grad(): 
            return model(x,tr)

    