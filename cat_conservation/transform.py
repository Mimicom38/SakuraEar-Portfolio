import torch
import torchvision
from torchvision import transforms

#画像サイズ224*224はResNetなどの一般的なモデルに合わせている
image_size = 224

#ImageNetの平均・標準偏差
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#データ変換
data_transform = {
  'train': transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.CenterCrop(image_size),
    #transforms.RandomHorizontalFlip(p=0.5), #p=0.5を追加
    #transforms.RandomRotation(5), #回転しすぎない
    #transforms.ColorJitter(
     #brightness=0.05, contrast=0.05 #0.5→0.1から変更
   # ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
  ]),

  'val': transforms.Compose([
    transforms.Resize(256), #224から変更
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)

  ])
}