# mimii-anomaly-detection
MIMII数据异常检测
基于自编码器的MIMII数据集机器声音异常检测，同时使用 -6dB、0dB、6dB 三组数据混合训练。

## 项目说明
- 模型：一维自编码器（AutoEncoder），利用重构误差判断音频是否异常
- 数据：MIMII 数据集，flac格式音频，包含 -6_dB、0_dB、6_dB 三个子文件夹
- 任务：正常/异常二分类，使用AUC作为评价指标
- 仓库包含：训练代码、评估AUC代码、训练完成权重 model.pkl

## 环境依赖
```bash
pip install torch librosa numpy scikit-learn
目录
MIMII_local/
├── -6_dB/
├── 0_dB/
├── 6_dB/
├── train.py        # 训练脚本，支持断点续训，训练结束保存最终权重model.pkl
├── eval_aucpy      # 评估脚本，计算AUC
└── model.pkl       # 训练完成的模型权重

训练命令
python3 train.py
