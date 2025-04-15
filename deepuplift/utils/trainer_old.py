import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



class EarlyStopper:
    # patience（默认值为 15）代表可以容忍验证损失连续不下降的轮次数量，
    # min_delta（默认值为 0）表示验证损失变化的最小差值（只有损失上升幅度超过该差值才会计数）
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
                t_pred,y_preds,*eps = self.model(X,tr)
                loss, outcome_loss, treatment_loss = self.loss_f(t_pred, y_preds,tr, y1, *eps)
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
                t_pred,y_preds, *eps = self.predict(X,tr)
                loss, outcome_loss, treatment_loss = self.loss_f(t_pred, y_preds,tr,y1, *eps)
                valid_loss.append(loss)
                valid_outcome_loss.append(outcome_loss)        
                valid_treatment_loss.append(treatment_loss)
        return torch.Tensor(valid_loss).mean(),  \
               torch.Tensor(valid_outcome_loss).mean(), \
               torch.Tensor(valid_treatment_loss).mean()

    def predict(self, x,tr=None):
        if isinstance(x, pd.DataFrame):  # 推理阶段: 如果x是df就转为numpy
            x = x.to_numpy()
        if tr is not None:
            tr = torch.Tensor(tr)
        x = torch.Tensor(x)
        with torch.no_grad():
            return self.model(x,tr)

