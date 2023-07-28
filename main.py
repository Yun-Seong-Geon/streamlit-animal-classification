from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from read_dataset import create_dataset
from train import preprocess
import tensorflow_hub as hub
import pandas as pd

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'], label='train loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='val loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='train accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.show()

data_cat_test,label_cat_test = create_dataset('dataset/test/cat',0)
data_dog_test,label_dog_test = create_dataset('dataset/test/dog',1)
data_wild_test,label_wild_test = create_dataset('dataset/test/wild',2)
data_dict = {
    'cat'  : data_cat_test,
    'dog'  : data_dog_test,
    'wild' : data_wild_test
}
labels_dict={
    'cat'  : label_cat_test,
    'dog'  : label_dog_test,
    'wild' : label_wild_test,
}      
X = []
Y = []

for key in data_dict.keys():
    X.extend(data_dict[key])
    Y.extend(labels_dict[key])
    
X = np.array(X)
Y = np.array(Y)

X,Y = preprocess(X,Y)
with keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = load_model('BigTransferModel.h5')

pred = model.predict(X)

pred = np.argmax(pred, axis=1)
plt.figure(figsize=(10, 10))


# 첫 9개의 이미지와 레이블을 출력합니다.
for i in range(9):
    # 3x3 그리드의 i+1번째 위치에 서브플롯을 추가합니다.
    ax = plt.subplot(3, 3, i + 1)
    
    # i번째 이미지를 출력합니다.
    plt.imshow(X[i])
    
    # 레이블을 출력합니다.
    plt.title(int(pred[i]))
    
    # 축 정보를 숨깁니다.
    plt.axis("off")
plt.savefig('predict_class.png')