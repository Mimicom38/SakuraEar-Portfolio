import cv2
import torch
import numpy as np
from PIL import Image
#model.pyからget_resnet18をインポート
from model import get_resnet18

#デバイスの設定(M4 Macなのでmps、次点でcuda、それ以外はcpu)
device = torch.device("mps" if torch.backends.mps.is_available()
else "cuda" if torch.cuda.is_available()
else "cpu")

print(f"使用デバイス: {device}")

#モデルの読み込みと重みの適用
model = get_resnet18()

#.pthで終わるファイル名に書き換える！
WEIGHT_FILE = "model.pth"

try:
  model.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
  print("モデルの重みを正常に読み込みました。")
except FileNotFoundError:
  print(f"{WEIGHT_FILE}が見つかりません。モデルを未学習の状態で起動")
  exit()

model.to(device)
model.eval() #推論モード
# 前処理の設定（一番シンプルな NumPy 方式に変更）
def safe_transform(pil_img):
  # 'val' の設定通り、256にリサイズして真ん中を224で切り抜く
   pil_img = pil_img.resize((256, 256))
   pil_img = pil_img.crop((16, 16, 240, 240))#224x224

 # PIL画像をNumPyの配列にして、0〜1に正規化
   img_np = np.array(pil_img, dtype=np.float32) / 255.0
#ここでPytorchのテンソルに変換
   tensor = torch.from_numpy(img_np)
#PytorchのモデルはCが先頭なのでチャンネルの順番を変更（HWC→CHW）して正規化
   tensor = tensor.permute(2, 0, 1)

    # ImageNetの平均(mean)と標準偏差(std)を手動で計算
   mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
   std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

   return (tensor - mean) / std
#判定用ラベル
labels = {0: "notsakura", 1:"sakura"}

#内蔵カメラの起動
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

print("リアルタイム判定を開始。終了するには 'q' キーを押してください。")

while True:
  ret, frame = cap.read()
  if not ret:
     print("フレームを取得できませんでした。")
     break

  #OpenCVのBGRからPILのRGBに変換
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(rgb_frame)

  #安全な前処理とテンソル化
  input_tensor = safe_transform(pil_img).unsqueeze(0).to(device)

  #予測
  with torch.no_grad():
     outputs = model(input_tensor)
     _, preds = torch.max(outputs, 1)
     prediction = preds.item()

#判定結果
  result_text = labels[prediction]

  #画面に判定結果を描画(左上に緑色で表示)
  cv2.putText(frame, result_text, (30, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

  #映像を表示
  cv2.imshow('sakura Detection', frame)

  #'q'キーで終了
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()