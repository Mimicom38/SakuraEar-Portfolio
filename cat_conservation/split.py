import os #ファイルやディレの操作が簡単に
import shutil #ファイルディレのコピー、移動、削除など
from sklearn.model_selection import train_test_split #機械学習では王道。自動シャッフル＆サイズ指定で分割

#保存先フォルダ
def split_and_copy(src_label):
  src_dir = src_label
  train_dir = f"dataset/train/{src_label}"
  val_dir = f"dataset/val/{src_label}"

  #なければ生成
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(val_dir, exist_ok=True) #あってもエラー回避

  all_images = [f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))]
  train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)#random_stateは同じ結果が出るように固定するためのシード値。42に意味はない(SF小説が元ネタらしい)
#train:val = 8:2
#シード値とは、乱数を固定するために使われる値(再現性高くしないと精度が乱れ比較できない。)
#all_imagesは画像ファイル名のリスト


  #コピー処理
  for img  in train_imgs:
    shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, img))
  #ん？元ディレも保存先ディレも一緒じゃない？一つ一つ画像コピーすんのかな？ファイルコピーだっけ？
  for img in val_imgs:
    shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, img))
  #os.path.joinで引数に渡した二つの文字列を結合し、一つのパスにできる。windowsなら\Macは/と合わせてくれる
  print(f"✅ {src_label} → Train: {len(train_imgs)}枚, Val: {len(val_imgs)}枚 コピー完了！")

#sakuraとnotsakuraの両方を処理
split_and_copy("sakura")
split_and_copy("notsakura")