import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def create_image_data_generator(dataframe, image_directory):
    dataframe['image_id'] = dataframe['image_id'].astype(str) + '.png'

    datagen = ImageDataGenerator(
        rescale = 1./255,  # Normalize the image
        width_shift_range = 0.2,  # Horizontal offset range
        height_shift_range = 0.2,  # Range of vertical offset
        shear_range = 0.2,  # Shear strength
        zoom_range = 0.2,  #Random scaling range
        brightness_range=[0.8, 1.2],  # Randomly adjust the range of image brightness
        fill_mode = 'nearest',  # Method to fill newly created pixels
        validation_split=0.2
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe = dataframe, 
        directory = image_directory, 
        x_col = 'image_id', 
        y_col = ['angle', 'speed'], 
        class_mode = 'raw',  # 因为这是一个回归问题，不是分类问题
        target_size = (320, 240),  # 假设我们希望所有图像都调整为320x240
        batch_size = 32,
        subset='training'
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=image_directory,
        x_col='image_id',
        y_col=['angle', 'speed'],
        class_mode='raw',
        target_size=(320, 240),
        batch_size=32,
        subset='validation' #Specify that this is verification data
    )

    return train_generator, validation_generator


def test_image_data_generator(generator, images_num = 5):

    images, labels = next(generator)

    for i in range(images_num):
        image_path = f'machine-learning-in-science-ii-2024/generator_test_images/image_{str(i+1)}.png'
        plt.imsave(image_path, images[i])
    

def main():
    data_path = 'machine-learning-in-science-ii-2024/training_norm.csv'
    image_directory = 'machine-learning-in-science-ii-2024/training_data/training_data'
    df = load_data(data_path)
    df = data_normalization(df)
    train_generator, validation_generator = create_image_data_generator(df, image_directory)


if __name__ == '__main__':
    main()
