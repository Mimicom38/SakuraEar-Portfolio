import torch
import torch.nn as nn  #ネットワークの構築
from torchvision.models import ResNet18_Weights
import torchvision.models as models

def get_resnet18():
  model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, 2)
  return model