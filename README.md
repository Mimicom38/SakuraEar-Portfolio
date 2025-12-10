# SakuraEar-Portfolio

# 概要（Overview）
本プロジェクトは、耳の形状（さくら耳）によってTNR(不妊去勢手術により野良猫の繁殖を抑え、数を減らす活動）済みか否かを判別する画像分類モデルです。  
TNRをした日本の野良猫は、耳をV字にカットされます。その耳がさくらのような形をしているため、カットを施された猫は「さくらねこ」と呼ばれています。
耳の切り方が浅い、角度が微妙、判断が困難な画像も含まれ、加えて画像の枚数も少なく精度に限界があります。

## 動機・背景
保護猫活動に強い関心があり、学生として直接現場に参加することが難しい私でも、
自身のスキルで少しでも力になれないかと思い立ち、制作に取り組みました。

*データセット作成について*

制作時に最も時間を費やしたのは、手作業による画像加工です。初めは200枚ずつ計400枚の画像を集めましたが作業時に誤って消去したため、次に計386枚の画像を集めました。左右反転やcanvaでの切り抜きで背景除去して試しましたが今度は精度が一向にあがりません。コードをSimpleCNNからResNet18に、transformの画像加工の過程を変更しても上がらなかったのですが、ChatGPTによると背景除去はこの細かな画像分類には不向きだとのことで、三度目の画像収集をしました。今度は顔まわり、耳を中心に背景を除去せず一枚一枚手作業で切り抜き、計520枚のデータセットを作りました。分類はsplitで自動分類し、左右反転もコードで行いましたが、それでも時間がかかりました。

*モデル構築　おわりに*

また、コーディングにつきましては、AIの理解度が初歩的な状態から始めたため、モデルの変更に伴うコードの変更や精度向上のための数値の変更など、様々な技術をGoogleやAIを使用し記述ごとにコツをつかむようにして制作しました。
しかし実務経験がないため、自身の技術がどれほど向上しているのかわかりません。勉強方法も模索しながら取り組みました。今後はより様々なプロジェクト制作に関わり、さらなる効率化に努めたいと思います。

## 技術スタック
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)   

![Static Badge](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)

![Static Badge](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

![Static Badge](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)
- torchvision

## 使用ツール
canva

## ファイル構成
.
├── __pycache__
│   ├── model.cpython-312.pyc
│   └── transform.cpython-312.pyc
├── dataset #学習に使用するデータセット
│   ├── train #訓練
│   └── val #検証
├── inference_test #推論に使用するサンプル画像
│   ├── catsample.jpg
│   ├── sakuramimi.jpeg
│   └── testcat.jpg
├── inference.py #推論
├── model.pth #学習済みモデルの保存
├── model.py #モデル構築
├── notsakura #未手術の猫画像
├── sakura #手術済みの猫画像
├── split.py #sakura/notsakuraを自動で分類
├── train.py #訓練
└── transform.py #データ加工

8 directories, 357 files(notsakura/sakuraの画像をのぞいて11files)

## 使用画像につきまして
※本モデルはインターネット上から収集した猫画像を使用し、学習目的のみで利用しています。 商用利用や再配布は行いません。 問題がある画像が含まれる場合は速やかに削除対応いたします。
以下は画像を使用させていただいたサイトのURLです。
[いのちつないだ♡ワンニャン写真・動画コンテスト]→
https://contest.doubutukikin.or.jp/gallery/sakura/20240715_11956.html
[iStock]→https://www.istockphoto.com/jp


## GoogleColabリンク

https://colab.research.google.com/drive/1bniTDI0aE2VLmnnrV2flWh-2cqd7z_B5?hl=ja#scrollTo=8tibzIJUH4ZM
