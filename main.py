import torch
from models import BertEncoder, CNNEncoder, FusionModel, Detector, optimizer
from utils.data_loader import get_dataloader
from models.trainer import train_model
from utils.evaluation import evaluate_model

class Args:
    def __init__(self):
        self.optim = 'adam'
        self.lr = 1e-4
        self.max_grad_norm = 1.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.warmup_steps = 10000
        self.visible_gpus = '0'

args = Args()

# 加载数据
train_loader = get_dataloader('data/preprocessed_data.pt', batch_size=32, shuffle=True)
val_loader = get_dataloader('data/preprocessed_data.pt', batch_size=32, shuffle=False)

# 初始化模型
bert_encoder = BertEncoder()
cnn_encoder = CNNEncoder()
fusion_model = FusionModel(text_dim=768, image_dim=2048, hidden_dim=512)
detector = Detector(input_dim=512)

# 构建模型
model = torch.nn.Sequential(
    bert_encoder,
    cnn_encoder,
    fusion_model,
    detector
)

# 构建优化器
optimizer = optimizer(args, model)

# 训练模型
trained_model = train_model(model, train_loader, optimizer, num_epochs=10)

# 评估模型
evaluate_model(trained_model, val_loader)
