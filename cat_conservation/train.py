from model import get_resnet18
from transform import data_transform
import torch
import torchvision
from tqdm import tqdm


train_dir = './dataset/train'
val_dir = './dataset/val'

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=data_transform['train'])
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=data_transform['val'])

#データをバッチ単位で読み込むDataLoaderを作成。
batch_size = 32
train_dataLoader = torch.utils.data.DataLoader(
  train_dataset, batch_size=batch_size, shuffle=True #訓練データをランダムにシャッフル
)
val_dataLoader = torch.utils.data.DataLoader(
  val_dataset, batch_size=batch_size, shuffle=False
)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu") #mac(mps)でもcuda gpuでも動きどちらでもなければcpuで動く

model = get_resnet18().to(device) #モデルを指定したデバイスへ移動

#　学習ループ
criterion = torch.nn.CrossEntropyLoss() #クロスエントロピー損失の定義
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #オプティマイザー定義

num_epochs = 30 #エポック数の定義

for epoch in range(num_epochs):
  running_loss = 0.0
  for images, labels in tqdm(train_dataLoader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    images = images.to(device) #画像と
    labels = labels.to(device) #ラベルを指定デバイスに移動させる
    
    optimizer.zero_grad() #勾配の初期化
    outputs = model(images) #推論
    loss = criterion(outputs, labels) 
    loss.backward() #逆伝播
    optimizer.step() #重み更新
    
    running_loss += loss.item() #各バッチのlossを+=して1エポックごとの合計Lossを出す
  
  print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

  #バリデーション
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in val_dataLoader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1) #引数1は行方向の最大値を求めるの意
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
  val_acc = correct / total * 100
  print(f"Val acc: {val_acc:2f}%") #バリデーションとval_accでどこがおかしいか確認する

# モデル保存　あとでモデルを再利用できる。
torch.save(model.state_dict(), 'model.pth')