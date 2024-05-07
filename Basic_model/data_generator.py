import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re, os
import cv2


def load_data(csv_path):
    return pd.read_csv(csv_path)
    #dataframe


def data_normalization(df):
    #Initialize a MinMaxScaler to normalize data to the [0, 1] range
    Scaler = MinMaxScaler()

    # Select the columns to be standardized
    columns_to_scale = ['angle', 'speed']

    # Standardize selected columns
    df[columns_to_scale] = Scaler.fit_transform(df[columns_to_scale])
    return df

##here need to move to data_preprocessing.py


datagen = ImageDataGenerator(
        rescale = 1./255,  # Normalize the image
        width_shift_range = 0.2,  # Horizontal offset range
        height_shift_range = 0.2,  # Range of vertical offset
        shear_range = 0.2,  # Shear strength
        zoom_range = 0.2,  #Random scaling range
        brightness_range=[0.6, 1.4],  # Randomly adjust the range of image brightness
        fill_mode = 'nearest',  # Method to fill newly created pixels
        validation_split=0.2
    )

def create_train_generator(dataframe, image_directory, target_size=(128, 128), batch_size=32):
    train_generator = datagen.flow_from_dataframe(
        dataframe = dataframe, 
        directory = image_directory, 
        x_col = 'image_id', 
        y_col = ['angle', 'speed'], 
        class_mode = 'raw',  # 因为这是一个回归问题，不是分类问题
        target_size = target_size,  # 假设我们希望所有图像都调整为320x240
        batch_size = batch_size,
        # subset='training'
    )

    return train_generator


def create_validation_generator(dataframe, image_directory, target_size=(128, 128), batch_size=32):
    validation_generator = datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = image_directory,
        x_col = 'image_id',
        y_col = ['angle', 'speed'],
        class_mode = 'raw',
        target_size = target_size,
        batch_size = batch_size,
        # subset = 'validation' #Specify that this is verification data
    )

    return validation_generator


def creat_train_generator_directory(train_dir, target_size = (128, 128), batch_size=32):
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  # or 'categorical', depending on your case
        subset='training'
    )
    return train_generator


def creat_validation_generator_directory(train_dir, target_size = (128, 128), batch_size=32):
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  # or 'categorical', depending on your case
        subset='validation'
    )
    return validation_generator


def create_train_generator_flow(images, labels, batch_size=32):
    datagen = datagen()
    train_generator = datagen.flow(
        images,
        labels,
        batch_size=batch_size
    )
    return train_generator


def create_validation_generator_flow(images, labels, batch_size=32):
    datagen = datagen()  # Usually, we don't apply augmentation to validation data
    validation_generator = datagen.flow(
        images,
        labels,
        batch_size=batch_size
    )
    return validation_generator


class ImageLabelGenerator(Sequence):
    def __init__(self, image_paths, angles, speeds, batch_size):
        self.image_paths = image_paths
        self.angles = angles
        self.speeds = speeds
        self.batch_size = batch_size

    #返回生成器能产生的总批次数
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_angle = self.angles[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_speed = self.speeds[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = np.array([img_to_array(load_img(file_path, target_size=(128, 128))) for file_path in batch_x])
        # 应用数据增强
        images = datagen.flow(images, batch_size=self.batch_size, shuffle=False).next()
        
        return images, [np.array(batch_y_angle), np.array(batch_y_speed)]


def test_image_data_generator(generator, images_num = 5):

    images, labels = next(generator)

    for i in range(images_num):
        image_path = f'machine-learning-in-science-ii-2024/generator_test_images/image_{str(i+1)}.png'
        plt.imsave(image_path, images[i])
    

def main():
    data_path = '/Users/mrs.zhuang/MLis2/training_norm.csv'
    image_directory = '/Users/mrs.zhuang/MLis2/training_data'
    # df = load_data(data_path)
    # df = data_normalization(df)
    # train_generator, validation_generator = create_image_data_generator(df, image_directory)
    


if __name__ == '__main__':
    main()
