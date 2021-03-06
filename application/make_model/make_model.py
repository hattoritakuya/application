# -*- coding: utf-8 -*-
"""make_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D-uHMoVRELDXU9yakT7BAQGsJyUaNsKy
"""

import tensorflow
import keras
print(tensorflow.__version__)
print(keras.__version__)

"""#訓練データの読み込み
##cifar-10の読み込み
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

"""##モデルの構築

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as np_utils

batch_size = 32
epochs = 20
n_class = 10

# one-hot表現に
y_train = np_utils.to_categorical(y_train,n_class)
y_test = np_utils.to_categorical(y_test,n_class)

model = Sequential()

# ゼロパディング、バッチサイズ以外の画像の形状を指定
model.add(Conv2D(32, (3,3), padding="same",input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #一次元の配列に変換
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5)) #ドロップアウト層
model.add(Dense(n_class)) 
model.add(Activation("softmax"))

model.compile(
    optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']
    )
model.summary()

"""##学習"""

#正規化
x_train = x_train / 255
x_test  = x_test / 255

generator = ImageDataGenerator(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True
)
generator.fit(x_train)

history = model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(x_test, y_test))

# 学習の推移を表示
import matplotlib.pyplot as plt

#訓練データの誤差
train_loss = history.history['loss']
#訓練データの精度
train_acc = history.history["accuracy"]
#検査データの誤差
val_loss = history.history['val_loss']
#訓練データの精度
val_acc = history.history["val_accuracy"]

plt.plot(np.arange(len(train_loss)), train_loss, label='loss')
plt.plot(np.arange(len(val_loss)), val_loss, label='val_loss')
plt.legend()
plt.show()

plt.plot(np.arange(len(train_acc)), train_acc, label='accuracy')
plt.plot(np.arange(len(val_acc)), val_acc, label='val_acc')
plt.legend()
plt.show()

# モデルの評価
loss, accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)

# モデルの保存とダウンロード
from google.colab import files

model.save("image_classifier.h5",include_optimizer=False)
files.download('image_classifier.h5')