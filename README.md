NewsGuard/
│
├── data/
│   ├── images/                # 存放图像数据
│   ├── texts/                 # 存放文本数据
│   ├── captions/              # 存放说明文字数据
│   └── preprocess.py          # 数据预处理脚本
│
├── models/
│   ├── __init__.py            # 模型初始化文件
│   ├── bert_encoder.py        # BERT编码器
│   ├── cnn_encoder.py         # CNN编码器
│   ├── fusion_model.py        # 多模态融合模型
│   ├── detector.py            # 检测模型
│   └── trainer.py             # 模型训练脚本
│
├── utils/
│   ├── data_loader.py         # 数据加载脚本
│   ├── evaluation.py          # 评估脚本
│   └── visualization.py       # 结果展示脚本
│
├── main.py                    # 主程序入口
└── requirements.txt           # 依赖包
