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
import requests
import time

FILE_ID = "1MN_hfzw78DVWT0JSPauIVLlzy7L4Wr2R"  # 여기에 모델 파일의 Google Drive ID를 입력하세요
MODEL_PATH = 'model.h5'
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def preprocess(image):
    image = tf.image.resize(image, [256,256]) / 255.0
    return image

def loads_model():
    # 파일이 로컬에 없는 경우에만 다운로드합니다
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
    
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict(dataes):
    Classes = {
    0:'cat',
    1:'dog',
    2:'wild'
    }
    with st.spinner('인공지능이 빵을 굽고 있어요...'):
        model = loads_model()  
    data_file = np.array([preprocess(data) for data in dataes])
    with st.spinner("인공지능 빵이 나오고 있어요...."):
        pred = model.predict(data_file)
        time.sleep(2)
    pred = np.argmax(pred,axis=1)

    if len(dataes) == 1:
        # Single image case
        fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
        ax.imshow(data_file[0])
        ax.set_title(Classes[pred[0]], fontsize=40)
        ax.axis("off")
        st.pyplot(fig) 
    else:
        # Multiple images case
        cols = math.ceil(math.sqrt(len(dataes)))
        rows = math.ceil(len(dataes) / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

        axs = axs.reshape(-1)

        for i, ax in enumerate(axs[:len(dataes)]):
            ax.imshow(data_file[i]) 
            ax.set_title(Classes[pred[i]], fontsize=40)
            ax.axis("off")
        
        if len(dataes) < len(axs):
            for ax in axs[len(dataes):]:
                fig.delaxes(ax)

        plt.tight_layout()
        st.pyplot(fig)
def main():
    st.title('Animals Classification')
    img_files = st.file_uploader('## 분류할 동물사진을 업로드 하세요.',type=['png','jpg','jpeg'],accept_multiple_files=True)
 
 
        
    if len(img_files) > 0:
        
        
        img_array = []
        for img_file in img_files:
            img = Image.open(img_file)
            img_array.append(np.array(img))
        try:
            predict(img_array)  
        except:
            pass

   
if __name__ == '__main__':
    
    main()
    