import os, re, csv, cv2
import pandas as pd
import numpy as np


def get_files_path(image_files_path):
    files_path_list = []
    filenames = sorted(os.listdir(image_files_path))
    for filename in filenames:
        if filename == '.DS_Store':
            continue
        else:
            files_path_list.append(os.path.join(image_files_path, filename))
    return files_path_list


def rename_image(image_files_path):
    counter = 0
    files_path_list = get_files_path(image_files_path)
    #define a regular rule to match the first part of a file nam
    pattern = re.compile(r'^[^_]+(_.+)$')
    for file_path in files_path_list:
        image_names = sorted(os.listdir(file_path))
        for image_name in image_names:
            if image_name.endswith('.png'):
                new_image_name = pattern.sub(f'{counter}\\1', image_name)
                if new_image_name != image_name:
                    new_path = os.path.join(file_path, new_image_name)
                    old_path = os.path.join(file_path, image_name)
                    os.rename(old_path, new_path)
                counter += 1
    print('Total:', counter+1)


def extract_data_from_filenames(image_files_path):
    files_path_list = get_files_path(image_files_path)
    with open('training_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'angle', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.png')

        for file_path in files_path_list:
            image_names = sorted(os.listdir(file_path))
            for image_name in image_names:
                if image_name.endswith('.png'):
                    match = pattern.match(image_name)
                    if match:
                        _, angle, speed = match.groups()
                        writer.writerow({'image_id':image_name, 'angle':angle, 'speed':speed})
    print("Finsh!")


def change_50(image_files_path):
    files_path_list = get_files_path(image_files_path)
    pattern = re.compile(r'^(\d+)_(\d+)_50\.png$')
    for file_path in files_path_list:
        image_names = sorted(os.listdir(file_path))
        for image_name in image_names:
            if image_name.endswith('.png'):
                match = pattern.match(image_name)
                if match:
                    new_image_name = f'{match.group(1)}_{match.group(2)}_35.png'
                    new_path = os.path.join(file_path, new_image_name)
                    old_path = os.path.join(file_path, image_name)
                    os.rename(old_path, new_path)
    print("finish")


def add_extension(file_path):
    df = pd.read_csv(file_path)
    df['image_id'] = df['image_id'].astype(str)
    df['image_id'] = df['image_id'].apply(lambda x: x if x.endswith('.png') else x + '.png')
    df.to_csv(file_path, index=False)


def extract_first_100_rows(input_csv, output_csv):
    df = pd.read_csv(input_csv, nrows=100)
    
    # 保存到新的CSV文件
    df.to_csv(output_csv, index=False)  # index=False表示不保存行索引到文件

    print(f"Data extracted and saved to {output_csv}")



def data_normalization(csv_file):
    df = pd.read_csv(csv_file)

    df['angle'] = (df['angle'] - 50) / 80
    df['speed'] = df['speed'] / 35

    df.to_csv('collect_data_norm.csv', index=False)


def main():
    image_files_path = 'training_data'
    file_path = '/home/alyzf6/MLis2/training_data.csv'
    # get_files_path(image_files_path)
    # rename_image(image_files_path)
    # extract_data_from_filenames(image_files_path)
    # change_50(image_files_path)
    # add_extension(file_path)
    # extract_first_100_rows(file_path, '100_train_norm.csv')
    data_normalization(file_path)


if __name__ == '__main__':
    main()