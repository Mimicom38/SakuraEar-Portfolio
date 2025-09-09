import torch
import torch.nn as nn  #ネットワークの構築
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
import torchvision.models as models

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2) 
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    self.fc1 = nn.Linear(32 * 56 * 56, 128)
    self.fc2 = nn.Linear(128, 2) #出力クラス数 ２クラスなので2
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x))) 
    x = self.pool(F.relu(self.conv2(x)))
    #最終的な特徴マップのサイズ
    print(x.shape)
    x = x.view(x.size(0), -1) #バッチサイズを固定して自動計算。-1は自動的にバッチサイズを計算するためのプレースホルダー
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
def get_resnet18():
  model = models.resnet18(weights=ResNet18_Weights.DEFAULT) #ResNet18のインスタンスを作成
  model.fc = nn.Linear(model.fc.in_features, 2) #出力クラス数2
  return model