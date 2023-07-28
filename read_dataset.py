from PIL import Image
import os
import numpy as np
import tensorflow as tf
import keras 

def create_dataset(image_path, label):
    # 빈 리스트를 생성하여 각 이미지와 레이블을 저장합니다.
    images = []
    labels = []

    if os.path.isdir(image_path):  # image_path가 디렉토리인 경우
        # 이미지 파일들의 이름을 모두 가져옵니다.
        image_files = os.listdir(image_path)

        for file in image_files:
            # 각 이미지 파일을 열고, 크기를 재조정하고, 픽셀 값을 numpy 배열로 변환합니다.
            image = Image.open(os.path.join(image_path, file))
            image = image.resize((128, 128))  # 이미지 크기를 재조정합니다. 필요한 크기로 조절하세요.
            image = np.array(image)

            # 이미지와 레이블을 각각의 리스트에 추가합니다.
            images.append(image)
            labels.append(label)

    elif os.path.isfile(image_path):  # image_path가 파일인 경우
        image = Image.open(image_path)
        image = image.resize((128, 128))  # 이미지 크기를 재조정합니다. 필요한 크기로 조절하세요.
        image = np.array(image)

        # 이미지와 레이블을 각각의 리스트에 추가합니다.
        images.append(image)
        labels.append(label)

    else:
        print(f"{image_path} is not a valid directory or file.")

    return images, labels

    # 데이터와 레이블 리스트를 numpy array로 변환합니다.
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

X = []
Y = []

# 디렉토리 생성
cat_images_directory = 'dataset/train/cat'
dog_images_directory = 'dataset/train/dog'
wild_images_directory = 'dataset/train/wild'
#함수를 사용하여 독립변수와 종속변수 분할
data_cat, labels_cat = create_dataset(cat_images_directory, 0)
data_dog, labels_dog = create_dataset(dog_images_directory, 1)
data_wild, labels_wild = create_dataset(wild_images_directory, 2)

# 분할 데이터 딕셔너리 생성
data_dict = {
    'cat'  : data_cat,
    'dog'  : data_dog,
    'wild' : data_wild
}
labels_dict={
    'cat'  : labels_cat,
    'dog'  : labels_dog,
    'wild' : labels_wild,
}             
X = []
Y = []

# 모든 카테고리를 순회하면서 이미지 데이터와 레이블을 병합합니다.
for key in data_dict.keys():
    X.extend(data_dict[key])
    Y.extend(labels_dict[key])

# 리스트를 numpy 배열로 변환합니다.
X = np.array(X)
Y = np.array(Y)

# 넘파이 배열을 파일로 저장합니다.
np.save('dataset/train/X.npy',X)
np.save('dataset/train/Y.npy',Y)
