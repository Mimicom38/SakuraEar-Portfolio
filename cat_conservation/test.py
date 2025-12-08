import os

train_notsakura = len(os.listdir("notsakura"))
train_sakura = len(os.listdir("sakura"))
val_notsakura = len(os.listdir("notsakura"))
val_sakura = len(os.listdir("sakura"))

print("train notsakura:", train_notsakura)
print("train sakura:", train_sakura)
print("val notsakura:", val_notsakura)
print("val sakura:", val_sakura)
