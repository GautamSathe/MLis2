import numpy as np
import pandas as pd
import os
import cv2
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Define function to extract number from filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1

# Define function to load and preprocess images
def load_images(image_folder):
    image_data = []
    filenames = os.listdir(image_folder)
    sorted_filenames = sorted(filenames, key=extract_number)
    for filename in sorted_filenames:
        if filename.endswith(".png"):
            print("Loading image:", filename)
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is None:
                print("Failed to load:", filename)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            image_data.append(img)
    return np.array(image_data)

# Load CSV file containing image filenames, steering angles, and speeds
csv_file_path = r"C:\Users\Gautam\Desktop\Mlis\training_norm.csv"
data_df = pd.read_csv(csv_file_path)

# Load and preprocess images
image_folder_path = r"C:\Users\Gautam\Desktop\Mlis\training_data"
images = load_images(image_folder_path)

# Split data into features and targets
X = images
y = data_df[['angle', 'speed']].values

# Initialize k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define a learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 15:
        lr *= 0.5
    elif epoch > 25:
        lr *= 0.1
    return lr

# Perform k-fold cross-validation
fold = 0
for train_indices, val_indices in kfold.split(X, y):
    fold += 1
    print(f"Fold: {fold}")
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # Augmentation for training data
    train_datagen = ImageDataGenerator(
        zoom_range=[0.8, 1.2],  # Randomly zoom images
        brightness_range=[0.6, 1.4]  # Randomly adjust brightness
    )

    # Apply augmentation to training data
    train_datagen.fit(X_train)
    
    # Load pre-trained models
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Freeze the pre-trained layers
    for base_model in [base_model_resnet, base_model_densenet, base_model_vgg]:
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom top layers for regression
    model_resnet = Sequential([
        base_model_resnet,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)
    ])

    model_densenet = Sequential([
        base_model_densenet,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)
    ])

    model_vgg = Sequential([
        base_model_vgg,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)
    ])

    # Compile the models with custom learning rate
    for model in [model_resnet, model_densenet, model_vgg]:
        model.compile(optimizer=Adam(learning_rate=lr_schedule(0)), loss='mse')

    # Define learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train the models with data augmentation and learning rate scheduling
    history_resnet = model_resnet.fit(train_datagen.flow(X_train, y_train, batch_size=128),
                                      validation_data=(X_val, y_val), epochs=15, callbacks=[lr_scheduler])
    history_densenet = model_densenet.fit(train_datagen.flow(X_train, y_train, batch_size=128),
                                          validation_data=(X_val, y_val), epochs=15, callbacks=[lr_scheduler])
    history_vgg = model_vgg.fit(train_datagen.flow(X_train, y_train, batch_size=128),
                                validation_data=(X_val, y_val), epochs=15, callbacks=[lr_scheduler])

    # Evaluate models
    loss_resnet = model_resnet.evaluate(X_val, y_val)
    loss_densenet = model_densenet.evaluate(X_val, y_val)
    loss_vgg = model_vgg.evaluate(X_val, y_val)

    print(f"Validation Loss - ResNet: {loss_resnet}, DenseNet: {loss_densenet}, VGG: {loss_vgg}")

    # Plot loss curves
    plt.figure(figsize=(12, 8))

    # Plot ResNet loss
    plt.subplot(3, 1, 1)
    plt.plot(history_resnet.history['loss'], label='ResNet Training Loss')
    plt.plot(history_resnet.history['val_loss'], label='ResNet Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet Training and Validation Loss')
    plt.legend()

    # Plot DenseNet loss
    plt.subplot(3, 1, 2)
    plt.plot(history_densenet.history['loss'], label='DenseNet Training Loss')
    plt.plot(history_densenet.history['val_loss'], label='DenseNet Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DenseNet Training and Validation Loss')
    plt.legend()

    # Plot VGG loss
    plt.subplot(3, 1, 3)
    plt.plot(history_vgg.history['loss'], label='VGG Training Loss')
    plt.plot(history_vgg.history['val_loss'], label='VGG Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG Training and Validation Loss')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'loss_curves_fold_{fold}.png')  # Save the loss plot as an image file

    # Testing the data
    test_image_folder_path = r"C:\Users\Gautam\Desktop\Mlis\test_data"
    test_images = load_images(test_image_folder_path)

    test_predictions_resnet = model_resnet.predict(test_images)
    test_predictions_densenet = model_densenet.predict(test_images)
    test_predictions_vgg = model_vgg.predict(test_images)

    ensemble_test_predictions = (test_predictions_resnet + test_predictions_densenet + test_predictions_vgg) / 3.0

    # Create a DataFrame with predictions and image IDs
    submission_df = pd.DataFrame(ensemble_test_predictions, columns=['angle', 'speed'])
    submission_df['image_id'] = range(1, len(test_images) + 1)

    # Reorder columns to match the required format
    submission_df = submission_df[['image_id', 'angle', 'speed']]

    # Save predictions to a CSV file
    submission_df.to_csv(f'submission_fold_{fold}.csv', index=False)