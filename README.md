# 📈 DeepUplift 项目介绍
DeepUplift 是一个基于深度学习实现异质性因果效果建模的项目，可用于解决工业界的权益补贴/商品定价/供需调节问题。本项目基于Pytorch框架，提供多种深度Uplift模型和评估指标，帮助社区更好地理解和应用深度因果模型。

## 🌟 功能特性
- 提供常用的深度Uplift模型：DragonNet, DragonDeepFM, EFIN, DESCN（完善中）
- 提供常用的评估指标：Qini 曲线和 AUUC 曲线（完善中）
- 提供数据处理PipLine与模型训练Trainer（完善中）

### 🔧 安装依赖
```bash
pip install pandas==2.1.4 sklearn==1.3.2 matplotlib==3.8.2 torch==1.12.1
```

## 🚀 快速开始
```python
from utils.evaluate import *
from trainer import Trainer
from models.DragonNet import *

# model
model = Trainer(model = DragonNet(num_features),epochs=2, batch_size=64)

# fit
model.fit(X_train, Y_train, T_train)

# predict
t_pred, y_preds = model.predict(X_test, T_test)

# evaluate
qini, qini_scores = uplift_metric(df, kind='qini')
```

## 🤝 贡献
如果你对本项目感兴趣，欢迎贡献代码、提出问题或建议。你可以通过提交 Pull Request 或 Issue 来参与开发。

## 📄 许可证
本项目采用 [MIT 许可证](LICENSE) 进行授权。
