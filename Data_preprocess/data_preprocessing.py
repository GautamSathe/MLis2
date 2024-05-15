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


def get_sorted_image_names(directory):
    files = os.listdir(directory)
    images = [file for file in files if file.endswith('.png')]
    images.sort()
    return images

def rename_image(image_files_path):
    counter = 0
    #define a regular rule to match the first part of a file nam
    pattern = re.compile(r'^[^_]+(_.+)$')

    image_names = get_sorted_image_names(image_files_path)
    for image_name in image_names:
        new_image_name = pattern.sub(f'{counter}\\1', image_name)
        new_path = os.path.join(image_files_path, new_image_name)
        old_path = os.path.join(image_files_path, image_name)
        os.rename(old_path, new_path)
        counter += 1
    print('Total:', counter)


def extract_data_from_filenames(image_files_path):
    with open('training_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'angle', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.png')

        image_names = get_sorted_image_names(image_files_path)
        for image_name in image_names:
            match = pattern.match(image_name)
            if match:
                image_id, angle, speed = match.groups()
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


def csv_merge(csv_file1, csv_file2):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv('combined.csv', index=False)


def change_name(csv_file):
    df = pd.read_csv(csv_file)
    df['image_id'] = df['image_id'].apply(lambda x: int(x.split('_')[0]))
    sorted_df = df.sort_values(by='image_id')
    sorted_df['image_id'] = sorted_df['image_id'].apply(lambda x: str(x) + '.png')
    sorted_df.to_csv('updated_csvfile.csv', index=False)


def rename_images1(images_file):
    counter = 0
    pattern = re.compile(r'\d+')
    images_name = sorted(os.listdir(images_file))
    for image_name in images_name:
        if image_name.endswith('png'):
            new_image_name = pattern.sub(f'{counter}', image_name)
            if new_image_name != image_name:
                new_path = os.path.join(images_file, new_image_name)
                old_path = os.path.join(images_file, image_name)
                os.rename(old_path, new_path)
                counter += 1

def recount_name(csv_file):
    df = pd.read_csv(csv_file)
    start_id = 0
    df['image_id'] = range(start_id, start_id + len(df))
    df['image_id'] = df['image_id'].apply(lambda x: str(x) + '.png')
    df.to_csv('updated_csvfile.csv', index=False)


def rename_images2(directory):
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png'): 
            new_name = filename.split('_')[0] + '.png'
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)

def fix_name(image_file):
    for filename in os.listdir(image_file):
        if filename.endswith('.png.png'):   
            new_name = filename[:-4]
            old_path = os.path.join(image_file, filename)
            new_path = os.path.join(image_file, new_name)
            os.rename(old_path, new_path)

def count_images(file_name):
    images = os.listdir(file_name)
    print(images)
    print(len(images))




def main():
    image_files_path = '/home/alyzf6/MLis2/collect_data'
    file_path = 'new_images_norm.csv'
    # get_files_path(image_files_path)
    # rename_image(image_files_path)
    # extract_data_from_filenames(image_files_path)
    # change_50(image_files_path)
    # add_extension(file_path)
    # extract_first_100_rows(file_path, '100_train_norm.csv')
    # data_normalization(file_path)
    csv_merge('collect_data_norm.csv','new_images_norm.csv')
    # change_name(file_path)
    # rename_images1(image_files_path)
    # recount_name(file_path)
    # rename_images2(image_files_path)
    # fix_name(image_files_path)
    # get_sorted_image_names(image_files_path)
    count_images(image_files_path)


if __name__ == '__main__':
    main()