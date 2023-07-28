import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.callbacks import EarlyStopping

X = np.load('dataset/train/X.npy')
Y = np.load('dataset/train/Y.npy')

trainX,valX,trainY,valY = train_test_split(X,Y,test_size=0.2,random_state=25)

"""
# 이미지를 출력할 큰 그림을 생성합니다.
plt.figure(figsize=(10, 10))


# 첫 9개의 이미지와 레이블을 출력합니다.
for i in range(9):
    # 3x3 그리드의 i+1번째 위치에 서브플롯을 추가합니다.
    ax = plt.subplot(3, 3, i + 1)
    
    # i번째 이미지를 출력합니다.
    plt.imshow(trainX[i])
    
    # 레이블을 출력합니다.
    plt.title(int(trainY[i]))
    
    # 축 정보를 숨깁니다.
    plt.axis("off")
plt.savefig('my_figure.png')
"""
def preprocess(image, label):
    image = tf.image.resize(image, [256,256]) / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY)).map(preprocess).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((valX, valY)).map(preprocess).batch(32)

model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_model = tf.keras.Sequential([hub.KerasLayer(model_url)])
num_classes = 3
bit_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

bit_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = bit_model.fit(train_ds, validation_data=val_ds,epochs=100,batch_size=32,callbacks=[early_stopping])
bit_model.save('BigTransferModel.h5')