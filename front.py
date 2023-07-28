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

import tensorflow as tf

def predict(dataes, model):
    Classes = {
    0:'cat',
    1:'dog',
    2:'wild'
    }
    
    data_file = np.array([preprocess(data) for data in dataes])
    
    pred = model.predict(data_file)
    pred = np.argmax(pred,axis=1)

    cols = math.ceil(math.sqrt(len(dataes)))
    rows = math.ceil(len(dataes) / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    axs = axs.reshape(-1)

    for i, ax in enumerate(axs[:len(dataes)]):
        # Use tf.cast instead of asType
        img = tf.cast(data_file[i]*255, tf.uint8)
        ax.imshow(img.numpy()) # convert tensor to numpy before plotting
        ax.set_title(Classes[pred[i]], fontsize=40)
        ax.axis("off")
    
    if len(dataes) < len(axs):
        for ax in axs[len(dataes):]:
            fig.delaxes(ax)

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.write('# Animals Classification')
    img_files = st.file_uploader('## 분류할 동물사진을 업로드 하세요.',type=['png','jpg','jpeg'],accept_multiple_files=True)
    stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
    if stream is not None:
        myzipfile = zipfile.ZipFile(stream)
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            root_folder = myzipfile.namelist()[0]
            model_dir = os.path.join(tmp_dir, root_folder)
            with keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
                model = tf.keras.models.load_model(model_dir)
        
    if img_files is not None:
        img_array = []
        for img_file in img_files:
            img = Image.open(img_file)
            img_array.append(np.array(img))
    
    if img_files is not None and stream is not None:
        predict(img_array,model)  
        
if __name__ == '__main__':
    main()
    