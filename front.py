import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import math
from PIL import Image
import zipfile
import tempfile
import os

def preprocess(image):
    image = tf.image.resize(image, [256,256]) / 255.0
    return image

def predict(data,model):
    Classes = {
    0:'cat',
    1:'dog',
    2:'wild'
    }

    
    data = preprocess(data)
    data = np.expand_dims(data, axis=0)
    
    pred = model.predict(data)
    pred = np.argmax(pred,axis=1)

    cols = math.ceil(math.sqrt(len(data)))
    # 필요한 행의 수를 계산
    rows = math.ceil(len(data) / cols)

    fig =plt.figure(figsize=(10, 10))
    for i in range(len(data)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow((data[i]*255).astype(np.uint8))
        plt.title(Classes[pred[i]],fontsize=40)
        plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    
    
def main():
    st.write('# Animals Classification')
    img_file = st.file_uploader('## 분류할 동물사진을 업로드 하세요.',type=['png','jpg','jpeg'])
    stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
    print(type(stream))
    if stream is not None:
        myzipfile = zipfile.ZipFile(stream)
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            root_folder = myzipfile.namelist()[0]
            model_dir = os.path.join(tmp_dir, root_folder)
            with keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
                model = tf.keras.models.load_model(model_dir)
        
    if img_file is not None:
        img = Image.open(img_file)
        img_array = np.array(img)
    
    if img_file is not None and stream is not None:
        predict(img,model)  
        
if __name__ == '__main__':
    main()
    