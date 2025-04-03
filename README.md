# 📈 DeepUplift 项目介绍
DeepUplift 是一个基于深度学习实现异质性因果效果建模的项目，基于Pytorch框架提供多种深度Uplift模型、评估指标和训练组件，帮助社区更好地理解和应用深度因果模型，解决权益补贴/商品定价/供需调节等工业界问题

## 🌟 功能特性
- Deep Uplift Models：✅DragonNet ✅DragonDeepFM ✅EFIN ✅DESCN ✅TarNet ✅CFRNet ✅EUEN >>> 🔍 
- Evaluate：✅Qini & AUUC  >>> 🛠
- Tools：✅Trainer ✅PSM  >>> 🛠

### 🔧 安装依赖
```bash
pip install pandas==2.1.4 sklearn==1.3.2 matplotlib==3.8.2 torch==1.12.1 geomloss==0.2.6
```

## 🚀 快速开始
```python
from utils.evaluate import *
from trainer import Trainer
from models.DESCN import *

# model
model = Trainer(model = ESX_Model(input_dim=len(features),share_dim=12,base_dim=12),
                epochs=20,batch_size=64,
                loss_f = partial(esx_loss))

# fit
model.fit(X_train, Y_train, T_train,valid_perc=0.2)

# predict
t_pred,y_preds, *_ = model.predict(X_test,T_test)

# evaluate
qini, qini_scores = uplift_metric(df, kind='qini')
```

## 🤝 贡献
如果你对本项目感兴趣，欢迎贡献代码、提出问题或建议。你可以通过提交 Pull Request 或 Issue 来参与开发。

## 📄 许可证
本项目采用 [MIT 许可证](LICENSE) 进行授权。
