- 项目概述
基于https://github.com/Zhuang-Zhuang-Liu/DeepUplift/tree/main，
扩展了多treatment训练支持及特征选择、embedding等模块，支持以下模型：
  - T-learner（已实现）
  - S-learner（已实现）
  - X-learner（已实现）
  - Tarnet（已实现）
  - ESN+Tarnet（已实现）
  - DESCN（已实现）
- 使用注意事项
  - 模型初始化：必须传入`treatment_label_list`参数（类型为`list`），用于定义treatment标签。其中：
    - `0`表示控制组
    - `1, 2, ..., n`表示不同treatment组
示例：`treatment_label_list=[0,1,2,3]`
  - 模型会直接输出ITE,前序传播输出的第一个参数即为ITE,使用list存储，每一个索引位置对应treatment_label_list中索引相应的treatment的预估lift
- 使用示例
  - main.ipynb：为训练示例，包含数据预处理、模型训练、模型评估等步骤
  - evaluate.ipynb：为模型评估示例，包含模型在测试集上的评估指标计算等步骤，多treatment评估结果
