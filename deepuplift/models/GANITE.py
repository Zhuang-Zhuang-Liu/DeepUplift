import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def parameter_setting_discriminator(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = False
    for params in discriminator.parameters():
        params.requires_grad = True
    for params in inference_net.parameters():
        params.requires_grad = False
    generator.eval()
    discriminator.train()
    inference_net.eval()

def parameter_setting_generator(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = True
    for params in discriminator.parameters():
        params.requires_grad = False
    for params in inference_net.parameters():
        params.requires_grad = False
    generator.train()
    discriminator.eval()
    inference_net.eval()

def parameter_setting_inference_net(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = False
    for params in discriminator.parameters():
        params.requires_grad = False
    for params in inference_net.parameters():
        params.requires_grad = True
    generator.eval()
    discriminator.eval()
    inference_net.train()


class GeneratorDeep(nn.Module):
    """Generator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels

    Returns:
      - G_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(GeneratorDeep, self).__init__()

        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim) # +2 for t and y
        self.dp1 = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList()
        for _ in range(k-3):
            layer = nn.Linear(h_dim, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            dp = nn.Dropout(p=0.2)
            self.layers.append(dp)

        # if t = 0, train fc31, 32
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # if t = 1, train fc41, 42
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)

    def forward(self, x, t, y):
        inputs = torch.cat([x, t, y], dim=1)
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            for layer in self.layers:
                x = layer(torch.relu(h1))
        else:
            h1 = torch.relu(self.fc1(inputs))
            for layer in self.layers:
                x = layer(torch.relu(h1))

        h31 = torch.relu(self.fc31(x))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)
        h41 = torch.relu(self.fc41(x))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)

class Discriminator(nn.Module):
    """Discriminator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels
      - hat_y: estimated counterfactuals

    Returns:
      - D_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Discriminator, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim) # +2 for t and y
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        nn.init.constant_(self.fc2_1.bias, 0)
        nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.constant_(self.fc2_2.bias, 0)

    def forward(self, x, t, y, hat_y):
        input0 = (1. - t) * y + t * hat_y[:, 0].unsqueeze(1) # if t = 0 dim=btx1
        input1 = t * y + (1. - t) * hat_y[:, 1].unsqueeze(1) # if t = 1
        inputs = torch.cat([x, input0, input1], dim=1)

        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))

        return self.fc3(h2)

class InferenceNetDeep(nn.Module):
    """Inference function.
    Args:
      - x: features
    Returns:
      - I_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(InferenceNetDeep, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.dp1 = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList()
        for _ in range(k-3):
            layer = nn.Linear(h_dim, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            dp = nn.Dropout(p=0.2)
            self.layers.append(dp)

        # Output: Estimated outcome when t = 0
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # Output: Estimated outcome when t = 1
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)

    def forward(self, x):
        inputs = x
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            for layer in self.layers:
                x = layer(torch.relu(h1))
        else:
            h1 = torch.relu(self.fc1(inputs))
            for layer in self.layers:
                x = layer(torch.relu(h1))

        h31 = torch.relu(self.fc31(x))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)
        h41 = torch.relu(self.fc41(x))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)




class GANITE:
    def __init__(self, input_dim, h_dim, is_self=True):
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.is_self = is_self
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # unit
        self.generator = GeneratorDeep(input_dim, h_dim, is_self, 15).to(self.device)
        self.discriminator = Discriminator(input_dim, h_dim, is_self).to(self.device)
        self.inference_net = InferenceNetDeep(input_dim, h_dim, is_self, 15).to(self.device)
        
        # optimizer
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0.9, 0.999))
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999))
        self.I_optimizer = optim.Adam(self.inference_net.parameters(), lr=1e-5, betas=(0.9, 0.999))


    def fit(self, X_train, Y_train, T_train, valid_perc=0.2, epochs=5, batch_size=64, learning_rate=1e-5, loss_f=None):
        # 转换为tensor
        X_train = torch.tensor(X_train.values if hasattr(X_train, 'values') else X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train.values if hasattr(Y_train, 'values') else Y_train, dtype=torch.float32).to(self.device)
        T_train = torch.tensor(T_train.values if hasattr(T_train, 'values') else T_train, dtype=torch.float32).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, T_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        for epoch in range(epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_i_loss = 0.0
            num_batches = 0
            
            for x, t, y in train_loader:
                t = t.unsqueeze(1)
                y = y.unsqueeze(1)
                
                # 1. 训练判别器
                parameter_setting_discriminator(self.generator, self.discriminator, self.inference_net)
                for _ in range(2):
                    y_tilde = self.generator(x, t, y)
                    d_logit = self.discriminator(x, t, y, y_tilde)
                    y_hat = self.inference_net(x)
                    D_loss, _, _ = ganite_loss(t, y,y_tilde,d_logit,y_hat)
                    self.D_optimizer.zero_grad()
                    D_loss.backward(retain_graph=True)
                    self.D_optimizer.step()
                
                # 2. 训练生成器
                parameter_setting_generator(self.generator, self.discriminator, self.inference_net)
                y_tilde = self.generator(x, t, y)
                d_logit = self.discriminator(x, t, y, y_tilde)
                y_hat = self.inference_net(x)
                _, G_loss, _ = ganite_loss(t, y,y_tilde,d_logit,y_hat)
                self.G_optimizer.zero_grad()
                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()
                
                # 3. 训练推理网络
                parameter_setting_inference_net(self.generator, self.discriminator, self.inference_net)
                y_tilde = self.generator(x, t, y)
                d_logit = self.discriminator(x, t, y, y_tilde)
                y_hat = self.inference_net(x)
                _, _, I_loss = ganite_loss(t, y,y_tilde,d_logit,y_hat)
                self.I_optimizer.zero_grad()
                I_loss.backward()
                self.I_optimizer.step()
                
                # 累加损失
                epoch_d_loss += D_loss.item()
                epoch_g_loss += G_loss.item()
                epoch_i_loss += I_loss.item()
                num_batches += 1
            
            # 计算平均损失并打印
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            avg_i_loss = epoch_i_loss / num_batches
            print(f"""--epoch: {epoch+1}
                  train丨discriminator_loss: {avg_d_loss:.4f} generator_loss: {avg_g_loss:.4f} inference_loss: {avg_i_loss:.4f}
                  """)


    def predict(self, x, Tr):
        x = torch.tensor(x.values if hasattr(x, 'values') else x, dtype=torch.float32).to(self.device)
        Tr = torch.tensor(Tr.values if hasattr(Tr, 'values') else Tr, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            y_hat = self.inference_net(x)
            y_hat = y_hat.cpu().numpy()
            
        return Tr.cpu().numpy(), [y_hat[:, 0], y_hat[:, 1]], None, None




def ganite_loss( t, y, y_tilde, d_logit, y_hat):
    """计算GANITE模型的损失函数
    
    Args:
        generator: 生成器模型
        discriminator: 判别器模型
        inference_net: 推理网络模型
        x: 特征
        t: 处理变量
        y: 观察到的结果
    
    Returns:
        tuple: (D_loss, G_loss, I_loss) 三个损失值
    """
    # 1. 计算判别器损失
    #y_tilde = generator(x, t, y)
    #d_logit = discriminator(x, t, y, y_tilde)
    D_loss = nn.BCEWithLogitsLoss()(d_logit, t)
    
    # 2. 计算生成器损失
    G_loss_GAN = -D_loss
    y_est = t * y_tilde[:, 1].view(-1, 1) + (1 - t) * y_tilde[:, 0].view(-1, 1)
    G_loss_factual = nn.BCEWithLogitsLoss()(y_est, y)
    G_loss = G_loss_factual + 1.0 * G_loss_GAN
    
    # 3. 计算推理网络损失
    #y_hat = inference_net(x)
    y_t0 = t * y + (1 - t) * y_tilde[:, 1].view(-1, 1)
    I_loss1 = nn.BCEWithLogitsLoss()(y_hat[:, 1].view(-1, 1), y_t0)
    y_t1 = (1 - t) * y + t * y_tilde[:, 0].view(-1, 1)
    I_loss2 = nn.BCEWithLogitsLoss()(y_hat[:, 0].view(-1, 1), y_t1)
    
    y_ate = torch.sum(t * y - (1 - t) * y)
    y_hat_ate = torch.sum(y_hat[:, 1] - y_hat[:, 0])
    supervised_loss = torch.nn.MSELoss()(y_hat_ate, y_ate)
    
    I_loss = I_loss1 + I_loss2 + 1.0 * supervised_loss
    
    return D_loss, G_loss, I_loss

