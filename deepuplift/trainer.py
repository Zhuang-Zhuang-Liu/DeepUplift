import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models.dragonnet import EarlyStopper


class Trainer:
    def __init__(
        self,
        model = None,
        epochs=200,
        batch_size=64,
        learning_rate=1e-5,
        data_loader_num_workers=4,
        loss_f = None,
        ):

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = data_loader_num_workers
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_dataloader = None
        self.valid_dataloader = None
        self.loss_f = loss_f

    def create_dataloaders(self, x, y, t, valid_perc=None):
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
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            x = torch.Tensor(x)
            t = torch.Tensor(t).reshape(-1, 1)
            y = torch.Tensor(y).reshape(-1, 1)
            train_dataset = TensorDataset(x, t, y)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )

    def fit(self, x, y, t, valid_perc=None):
        self.create_dataloaders(x, y, t, valid_perc)
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        for epoch in range(self.epochs):
            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                if self.model.__class__.__name__ == 'EFIN':
                    y0_pred, y1_pred, t_pred, eps = self.model(X, tr, y1)
                    loss,loss0,loss1,loss_t = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
                elif self.model.__class__.__name__ == 'DragonNet':
                    t_pred,y_preds, eps = self.model(X)
                    loss, outcome_loss, treatment_loss = self.loss_f(t_pred, y_preds, eps, tr, y1)
                elif self.model.__class__.__name__ == 'DragonNetDeepFM': 
                    t_pred,y_preds = self.model(X)
                    loss, outcome_loss, treatment_loss = self.loss_f(t_pred,y_preds,tr,y1)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if self.valid_dataloader:
                self.model.eval()
                valid_loss,valid_outcome_loss,valid_treatment_loss = self.validate_step()
                print(f"""--epoch: {epoch} 
                train_loss: {loss:.4f} outcome_loss:{outcome_loss:.4f} treatment_loss:{treatment_loss:.4f} 
                valid_loss: {valid_loss:.4f} valid_outcome_loss:{valid_outcome_loss:.4f} valid_treatment_loss:{valid_treatment_loss:.4f} """
                )
                self.model.train()
                if early_stopper.early_stop(valid_loss):
                    break
            else:
                print(f"""epoch: {epoch} 
                train_loss: {loss:.4f} outcome_loss:{outcome_loss:.4f} treatment_loss:{treatment_loss:.4f} """)

    def validate_step(self):
        valid_loss = []
        valid_outcome_loss = []
        valid_treatment_loss = []
        with torch.no_grad():
            for batch, (X, tr, y1) in enumerate(self.valid_dataloader):
                if self.model.__class__.__name__ == 'EFIN':
                    y0_pred, y1_pred, t_pred, eps = self.model(X, tr, y1)
                elif self.model.__class__.__name__ == 'DragonNet' :
                    t_pred,y_preds, eps = self.predict(X)
                    loss, outcome_loss, treatment_loss = self.loss_f(t_pred, y_preds, eps,tr,y1)
                elif self.model.__class__.__name__ == 'DragonNetDeepFM':
                    t_pred,y_preds = self.model(X)
                    loss, outcome_loss, treatment_loss = self.loss_f(t_pred,y_preds,tr,y1)

                valid_loss.append(loss)
                valid_outcome_loss.append(outcome_loss)        
                valid_treatment_loss.append(treatment_loss)
        return torch.Tensor(valid_loss).mean(),  \
               torch.Tensor(valid_outcome_loss).mean(), \
               torch.Tensor(valid_treatment_loss).mean()

    def predict(self, x):
        if isinstance(x, pd.DataFrame):  # 推理阶段: 如果x是df就转为numpy
            x = x.to_numpy()
        x = torch.Tensor(x)
        with torch.no_grad():
            return self.model(x)

