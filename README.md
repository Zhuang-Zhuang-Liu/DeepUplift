# 📈 DeepUplift 项目介绍
DeepUplift 是一个基于深度学习实现异质性因果效果建模的项目，基于Pytorch框架提供多种深度Uplift模型、评估指标和训练组件，帮助社区更好地理解和应用深度因果模型，解决权益补贴/商品定价/供需调节等工业界问题

## 🌟 功能特性
- Deep Uplift Models：
  - ✅ DragonNet：基于深度学习的因果推断模型
  - ✅ DragonDeepFM：结合DeepFM的因果推断模型
  - ✅ EFIN：基于特征交互的因果推断网络
  - ✅ DESCN：深度因果效应网络
  - ✅ TarNet：目标网络模型
  - ✅ CFRNet：反事实回归网络
  - ✅ EUEN：端到端因果推断网络
  - 🔄 GANITE：开发中
  - 🔄 CEVAE：开发中
  - 🔄 EEUEN：开发中

- 评估指标：
  - ✅ Qini/AUUC 曲线
  - ✅ 因果效应评估指标
  - ✅ 模型性能评估工具

- 工具支持：
  - ✅ 模型训练器
  - ✅ 倾向得分匹配(PSM)
  - ✅ TensorBoard 可视化
  - ✅ 数据预处理工具

### 🔧 安装依赖
```bash
pip install pandas==2.1.4 sklearn==1.3.2 matplotlib==3.8.2 torch==1.12.1 geomloss==0.2.6
```

## 🚀 快速开始
```python
from deepuplift.models import DESCN
from deepuplift.utils.evaluate import uplift_metric

# 初始化模型
model = DESCN.ESX(input_dim=len(features), share_dim=12, base_dim=12)
loss_f = partial(esx_loss)

# 模型训练
model.fit(
    X_train, 
    Y_train, 
    T_train,
    valid_perc=0.2,
    epochs=2,
    batch_size=64,
    learning_rate=1e-5,
    loss_f=loss_f,
    tensorboard=True
)

# 模型预测
t_pred, y_preds, *_ = model.predict(X_test, T_test)

# 模型评估
qini, qini_scores = uplift_metric(df, kind='qini')
```

## 📁 项目结构
```
deepuplift/
├── models/          # 模型实现
├── utils/           # 工具函数
│   ├── evaluate.py  # 模型指标
│   ├── matrics.py   # loss指标
│   └── psm.py       # 倾向得分匹配
└── main.py          # 主程序入口
```

## 🤝 贡献
如果你对本项目感兴趣，欢迎贡献代码、提出问题或建议。你可以通过提交 Pull Request 或 Issue 来参与开发。

## 📄 许可证
本项目采用 [MIT 许可证](LICENSE) 进行授权。
