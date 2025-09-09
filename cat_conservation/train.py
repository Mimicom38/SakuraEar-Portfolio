from model import SimpleCNN
from transform import data_transform
import torch
import torchvision
from tqdm import tqdm

#　データの読み込み(ディレ指定)
train_dir = './dataset/train'
val_dir = './dataset/val'
#データセットの作成(指定ディレから画像を読み込み、指定した前処理を適用)
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=data_transform['train'])
val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=data_transform['val'])

#データをバッチ単位で読み込むDataLoaderを作成。
batch_size = 32
train_dataLoader = torch.utils.data.DataLoader(
  train_dataset, batch_size=batch_size, shuffle=True #トレデータをランダムにシャッフル
)
val_dataLoader = torch.utils.data.DataLoader(
  val_dataset, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device) #モデルを指定したデバイスへ移動

#　学習ループ
num_epochs = 10 #エポック数の定義
criterion = torch.nn.CrossEntropyLoss() #クロスエントロピー損失の定義
optimizer = torch.optim.Adam(model.parameters()) #オプティマイザー定義

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


# モデル保存　あとでモデルを再利用できる。
torch.save(model.state_dict(), 'model.pth')

