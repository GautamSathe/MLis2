import cv2, os
import pandas as pd
import numpy as np
import matplotlib as plt
from data_generator import create_train_generator, create_validation_generator, load_data
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import KFold


def test_image_generator(full_df, directory, k = 5, target_size=(128, 128), batch_size=1):
    """
    测试ImageDataGenerator使用flow_from_dataframe方法能否正常读取图像。
    
    参数:
    dataframe (pd.DataFrame): 包含图像文件路径和标签的DataFrame。
    directory (str): 图像文件的根目录路径。
    target_size (tuple): 每个输出图像的尺寸。
    batch_size (int): 每批处理的图像数量。
    """
    
    # 创建ImageDataGenerator实例
    datagen = ImageDataGenerator()
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold,(train_idx, val_idx) in enumerate(kfold.split(full_df), 1):
        print(f"Fold:{fold}/{k}")
        print(f'num_train:{len(train_idx)}, num_val:{len(val_idx)}, num_total:{len(train_idx)+len(val_idx)}')

        df_train = full_df.iloc[train_idx]
        df_val = full_df.iloc[val_idx]


        # 创建生成器
        train_generator = datagen.flow_from_dataframe(
            dataframe = df_train, 
            directory = directory, 
            x_col = 'image_id', 
            y_col = ['angle', 'speed'], 
            class_mode = 'raw',  # 因为这是一个回归问题，不是分类问题
            target_size = target_size,  # 假设我们希望所有图像都调整为320x240
            batch_size = batch_size,
            # subset='training'
        )

        validation_generator = datagen.flow_from_dataframe(
            dataframe = df_val,
            directory = directory,
            x_col = 'image_id',
            y_col = ['angle', 'speed'],
            class_mode = 'raw',
            target_size = target_size,
            batch_size = batch_size,
            # subset = 'validation' #Specify that this is verification data
        )

    
    # # 初始化统计
    # success_count = 0
    # fail_count = 0
    # failed_images = []

    # # 遍历所有图像
    # for i in range(len(dataframe)):
    #     try:
    #         # 加载图像
    #         img, label = next(train_generator)
    #         success_count += 1
    #     except Exception as e:
    #         # 记录失败的加载尝试
    #         print(f"Failed to process image: {dataframe.iloc[i]['filename']} - Error: {e}")
    #         failed_images.append(dataframe.iloc[i]['filename'])
    #         fail_count += 1

    # print(f"Total images processed successfully: {success_count}")
    # print(f"Total images failed to process: {fail_count}")
    # if fail_count > 0:
    #     print("Failed images:", failed_images)



def main():
    csv_file = 'training_norm.csv'
    images_file = 'training_data'

    
    df = load_data(csv_file)
    test_image_generator(df, images_file)


if __name__ == '__main__':
    main()