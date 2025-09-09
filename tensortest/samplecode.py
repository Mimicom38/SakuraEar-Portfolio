import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

#mnistデータセット読み込み
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#画像データの正規化
train_images = train_images / 255.0
test_images = test_images /255.0

#モデル構築
model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation="relu"),
  Dense(10, activation="softmax")
])

#モデルのコンパイル
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

#モデルのトレーニング
model.fit(train_images, train_labels, epochs=5)

#モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")