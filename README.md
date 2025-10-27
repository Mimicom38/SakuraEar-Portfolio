# SakuraSensor-Portfolio

# 概要（Overview）
本プロジェクトは、耳の形状（さくら耳）によってTNR(不妊去勢手術により野良猫の繁殖を抑え、数を減らす活動）済みか否かを判別する画像分類モデルです。  
TNRをした日本の野良猫は、耳をV字にカットされます。その耳がさくらのような形をしているため、カットを施された猫は「さくらねこ」と呼ばれています。

耳の切り方が浅い、角度が微妙、判断が困難な画像も一部含まれていますが、これは現実のTNR現場でも見られる課題であり、学習モデルの汎用性と柔軟性向上のために採用しています。 ただし、画像枚数が少なく精度に限界があります。

## 動機・背景
保護猫活動に強い関心があり、自分にも何かできないかと考えておりました。
しかし学生である私は時間やお金の制約もあり、現場で直接支援活動をすることが難しい状況にあります。
そういったなかで、自分が学んでいるプログラミングを使って少しでも現場の助けになるものを作れないかと思い、ポートフォリオ制作に取り組みました。

## 技術スタック
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)   

![Static Badge](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)

![Static Badge](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

![Static Badge](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)
- torchvision
- PIL (画像処理)

## ファイル構成

## 使用画像につきまして
※本モデルはインターネット上から収集した猫画像を使用し、学習目的のみで利用しています。 商用利用や再配布は行いません。 問題がある画像が含まれる場合は速やかに削除対応いたします。
以下は画像を使用させていただいたサイトのURLです。
[いのちつないだ♡ワンニャン写真・動画コンテスト]→
https://contest.doubutukikin.or.jp/gallery/sakura/20240715_11956.html
[iStock]→https://www.istockphoto.com/jp



## Colabリンク
https://colab.research.google.com/drive/1bniTDI0aE2VLmnnrV2flWh-2cqd7z_B5?usp=sharing

## GitHubリポジトリ
