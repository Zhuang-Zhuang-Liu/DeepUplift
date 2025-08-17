import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
import torch
from torch.utils.data import Dataset
import random

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_dataloader = None
        self.valid_dataloader = None

    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def save_checkpoint(self,model, optimizer, epoch, filepath='checkpoint.pth'):
        base, ext = filepath.rsplit('.', 1)  # 分割文件名和扩展名
        new_filepath = f"{base}_{epoch}.{ext}"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, new_filepath)

    def load_checkpoint(self,model, optimizer, filepath='checkpoint.pth', device='cpu'):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return model, optimizer, start_epoch

    def create_dataloaders(self, df,feature_list,discrete_cols,batch_size=64,num_workers=4, pin_memory=True,valid_perc=False,label_y=None,label_treatment=None):
            continuous_cols = [each for each in feature_list if each not in discrete_cols]
            self.set_seed(42)
            if valid_perc:
                df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
                X_train_discrete = torch.tensor(df_train[discrete_cols].values, dtype=torch.float32)
                X_train_continuous = torch.tensor(df_train[continuous_cols].values, dtype=torch.float32)
                t_train = torch.tensor(df_train[label_treatment].values, dtype=torch.float32).unsqueeze(1)
                print(df_train[label_y].values)
                y_train = torch.tensor(df_train[label_y].values, dtype=torch.float32).unsqueeze(1)
                X_train = torch.cat((X_train_continuous, X_train_discrete), dim=1) 

                X_test = torch.tensor(df_test[feature_list].values, dtype=torch.float32)
                X_test_discrete = torch.tensor(df_test[discrete_cols].values, dtype=torch.float32)
                X_test_continuous = torch.tensor(df_test[continuous_cols].values, dtype=torch.float32)
                t_test = torch.tensor(df_test[label_treatment].values, dtype=torch.float32).unsqueeze(1)
                y_test = torch.tensor(df_test[label_y].values, dtype=torch.float32).unsqueeze(1)

                self.train_dataloader = DataLoader(TensorDataset(X_train, t_train, y_train, X_train_discrete, X_train_continuous), batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False)
                self.valid_dataloader = DataLoader(TensorDataset(X_test, t_test, y_test, X_test_discrete, X_test_continuous), batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False)
                print("预计单epoch训练步数:",int(df_train.shape[0]/batch_size))
            else:
                X_discrete = torch.tensor(df[discrete_cols].values, dtype=torch.float32)
                X_continuous = torch.tensor(df[continuous_cols].values, dtype=torch.float32)
                t = torch.tensor(df[label_treatment].values, dtype=torch.float32).unsqueeze(1)
                y = torch.tensor(df[label_y].values, dtype=torch.float32).unsqueeze(1)
                X = torch.cat((X_continuous, X_discrete), dim=1) 

                dataset = TensorDataset(X, t, y,X_discrete,X_continuous)
                
                self.train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False)
                print("预计单epoch训练步数:",int(df.shape[0]/batch_size))

    def fit(self, df,feature_list,discrete_cols,batch_size=64, epochs=10,
            learning_rate=1e-5,uplift_loss_f=None,loss_f=None,loss_f_eps=(), tensorboard=False,num_workers=4,pin_memory=True,task=None,device=None, valid_perc=False,label_y=None,label_treatment=None,loss_type=None,classi_nums=2, treatment_label_list=None,checkpoint_path=None,if_continued_train=0
            ):

        self.set_seed(42)
        model = self.train()
        # model = nn.DataParallel(model)  # 包装模型，使用所有可用GPU
        model = model.to(device)         
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if if_continued_train:
            model, optim, start_epoch = self.load_checkpoint(model, optim, checkpoint_path, device)
            model.train()
            epochs = epochs - start_epoch
            print(f'剩余{epochs}')
        
        self.create_dataloaders(df=df,feature_list=feature_list, discrete_cols=discrete_cols,batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory,valid_perc=valid_perc,label_y=label_y,label_treatment=label_treatment)
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=5, factor=0.1, verbose=True)

        for epoch in range(epochs):
            import time
            torch.cuda.synchronize()
            start_time = time.time()
            loss_ = []
            for batch, (X, tr, y1, X_discrete, X_continuous) in enumerate(self.train_dataloader):
                # print(batch)
                # torch.cuda.synchronize()
                # start_time = time.time()
                
                optim.zero_grad()
            
                X = X.to(device)
                tr = tr.to(device)
                y1 = y1.to(device)
                X_discrete = X_discrete.to(device)
                X_continuous = X_continuous.to(device)

                # end_time = time.time()
                # print(f"epoch time: {end_time - start_time:.4f}s")
                
                # start_time = time.time()
                t_pred,y_preds,*eps = model(X,tr,X_discrete, X_continuous)
                # end_time = time.time()
                # print(f"epoch time: {end_time - start_time:.4f}s")

                # print(f"back time: {time.time() - start:.4f}s")
                # start_time = time.time()
                loss, loss_control, loss_treat = loss_f(y_preds,tr, y1,task,loss_type,classi_nums, treatment_label_list)
                # end_time = time.time()
                # print(f"epoch time: {end_time - start_time:.4f}s")
        
                # start_time = time.time()
                loss.backward()  # 这里包裹反向传播
                # end_time = time.time()
                # print(f"epoch time: {end_time - start_time:.4f}s")

                # start_time = time.time()
                optim.step()
                # end_time = time.time()
                # print(f"epoch time--------------: {end_time - start_time:.4f}s")
                # print(i,outcome_loss)

                loss_.append(loss.item())

                # print(f"back time: {time.time() - start:.4f}s")

            # print(f"epoch time--------------: {time.time() - start_time:.4f}s")  

            # start_time = time.time()
            if self.valid_dataloader:
                model.eval()
                # if isinstance(model, nn.DataParallel):

                loss_valid = []
                treatment_valid = []
                for batch_valid, (X_valid, tr_valid, y1_valid, X_discrete_valid, X_continuous_valid) in enumerate(self.valid_dataloader):
                    X_valid = X_valid.to(device)
                    tr_valid = tr_valid.to(device)
                    y1_valid = y1_valid.to(device)
                    X_discrete_valid = X_discrete_valid.to(device)
                    X_continuous_valid = X_continuous_valid.to(device)
                    
                    t_pred,y_preds,*eps = self.predict(X_valid,tr_valid,device,X_discrete_valid, X_continuous_valid)

                    loss, outcome_loss, treatment_loss = loss_f(y_preds,tr_valid, y1_valid,task,loss_type,classi_nums, treatment_label_list)
                    if uplift_loss_f:
                        treatment_loss = uplift_loss_f(t_pred, y_preds,tr_valid, y1_valid, *loss_f_eps)
                        treatment_valid.append(treatment_loss)
                    # y_preds = [torch.expm1(y_pred) for y_pred in y_preds]
                    loss_valid.append(loss.item())

                print(f"""--epoch: {epoch} train_loss: {np.mean(loss_):.4f}  valid_loss: {np.mean(loss_valid):.4f} uplift_loss: {np.mean(treatment_valid):.4f} """)

                # scheduler.step(np.mean(loss_valid))
                
                model.train()

                if early_stopper.early_stop(np.mean(loss_valid)):
                    break
            else:
                print(f"""epoch: {epoch} train_loss: {np.mean(loss_):.4f}""")
            # print(f"epoch: {epoch} time: {time.time() - start_time:.4f}s")

            # start_time = time.time()
            self.save_checkpoint(model, optim, epoch, checkpoint_path)
            print(f"epoch: {epoch} time: {time.time() - start_time:.4f}s")

    def predict(self, x,tr=None,device=None,X_discrete=None, X_continuous=None):
        model = self.eval()
        if isinstance(x, pd.DataFrame):  # 推理阶段: 如果x是df就转为numpy
            x = x.to_numpy()
            x = torch.Tensor(x)
        if tr is not None:
            if isinstance(tr, pd.Series):
                tr = tr.to_numpy()
                tr = torch.Tensor(tr).to(device)
                
        x = x.to(device)
        with torch.no_grad(): 
            return model(x,tr,X_discrete, X_continuous)


