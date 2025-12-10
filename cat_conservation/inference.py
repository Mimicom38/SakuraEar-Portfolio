import torch
import transform
from torchvision import transforms, datasets
from PIL import Image
from model import get_resnet18

#モデルの読み込み
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

#学習したモデルの構造と重みを読み込み
model = get_resnet18().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval() #推論モード

train_dataset = datasets.ImageFolder('./dataset/train')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
print("class mapping:", idx_to_class)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#transform.pyと同じtransformを使う
infer_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  # transforms.ColorJitter(
  #   brightness=0.2,
  #   contrast=0.2
  # ),
  transforms.ToTensor(),
  transforms.Normalize(mean, std)
])

#画像の読み込みと前処理
image_path = "inference_test/nosakura.jpg"
image = Image.open(image_path).convert("RGB")
#バッチ次元を追加
tensor = infer_transform(image).unsqueeze(0).to(device)

#推論
with torch.no_grad():
  outputs = model(tensor) #image→tensorにした。 #logits
  probs = torch.nn.functional.softmax(outputs, dim=1) #probs追加
  _, predicted = torch.max(outputs, 1)

  print(outputs)

# 4. 結果表示
class_names = ["notsakura", "sakura"]
print(f"この猫はたぶん{class_names[predicted.item()]} です")

print("raw logits:", outputs)
print("probs:", probs)
print("predicted:", predicted.item(), "predicted class:", idx_to_class[predicted.item()])