# 📈 DeepUplift 
DeepUplift is a project that implements heterogeneous causal effect modeling based on deep learning. It provides various deep Uplift models, evaluation metrics, and training components based on the PyTorch framework, helping the community better understand and apply deep causal models to solve industrial problems such as equity subsidies, product pricing, and supply-demand regulation.

## 🌟 Features
- Deep Uplift Models：
  - ✅ TarNet: Estimating individual treatment effect: generalization bounds and algorithms, 2016.
  - ✅ CFRNet: Estimating individual treatment effect: generalization bounds and algorithms, 2016.
  - ✅ CEVAE: Causal effect inference with deep latent-variable models. In Advances in Neural Information Processing Systems, 2017.
  - ✅ GANITE：Estimation of Individualized Treatment Effects using Generative Adversarial Nets, International Conference on Learning Representations (ICLR), 2018.
  - ✅ DragonNet: Adapting Neural Networks for the Estimation of Treatment Effects, 2019.
  - ✅ DragonDeepFM: Adapting Neural Networks for the Estimation of Treatment Effects, 2019.
  - ✅ EUEN: Addressing Exposure Bias in Uplift Modeling forLarge-scale Online Advertising, IEEE International Conference on Data Mining (ICDM), 2021.
  - ✅ EEUEN: Addressing Exposure Bias in Uplift Modeling forLarge-scale Online Advertising, IEEE International Conference on Data Mining (ICDM), 2021.
  - ✅ DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation, SIGKDD, 2022.
  - ✅ EFIN: Explicit Feature Interaction-aware Uplift Network for Online Marketing, SIGKDD, 2023.
  - 🔄 SNet: Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms, 2021

- Evaluation Metrics：
  - ✅ Qini/AUUC Curves
  - ✅ Causal Effect Evaluation Metrics
  - ✅ Model Performance Evaluation Tools

- Tool Support：
  - ✅ Model Trainer
  - ✅ Propensity Score Matching (PSM)
  - ✅ TensorBoard Visualization
  - ✅ Data Preprocessing Tools

### 🔧 Installation Dependencies
```bash
pip install pandas==2.1.4 sklearn==1.3.2 matplotlib==3.8.2 torch==1.12.1 geomloss==0.2.6
```

## 🚀 Quick Start
```python
from deepuplift.models.DESCN import ESX
from deepuplift.utils.evaluate import uplift_metric

# Initialize model
model = ESX(input_dim=len(features), share_dim=12, base_dim=12)
loss_f = partial(esx_loss)

# Model training
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

# Model prediction
t_pred, y_preds, *_ = model.predict(X_test, T_test)

# Model evaluation
qini, qini_scores = uplift_metric(df, kind='qini')
```

## 📁 Project Structure
```
deepuplift/
├── models/          # Model implementations
├── utils/           # Utility functions
│   ├── evaluate.py  # Model metrics
│   ├── matrics.py   # Loss metrics
│   └── psm.py       # Propensity score matching
└── main.py          # Main program entry
```

## 🤝 Contribution
If you are interested in this project, you are welcome to contribute code, raise issues, or make suggestions. You can participate in development by submitting Pull Requests or Issues.

## 💬 Contact Me
💚 微信: Wave_1024
📚 公众号: 壮壮的三味书屋
![image](https://github.com/user-attachments/assets/defbb3a5-10c4-4288-8555-eef1631dfa0d?width=40)

## 📄 License
This project is licensed under the [MIT License](LICENSE).
