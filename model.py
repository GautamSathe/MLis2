import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.applications import VGG19, DenseNet121, ResNet50
from keras.callbacks import Callback, EarlyStopping
from data_generator import create_train_generator, create_validation_generator


def load_data(csv_path):
    return pd.read_csv(csv_path)
    #dataframe


def load_images(csv_file, images_file):
    images = []
    csv_data = pd.read_csv(csv_file)
    image_directory = images_file
    image_paths = csv_data['image_id'].apply(lambda x: os.path.join(image_directory, x)).tolist()
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print('Failed to loaded:', image_path)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img.reshape((1,) + img.shape)
        images.append(img_array)
        print("append:",image_path)

    np_images = np.array(images)
    return np_images
    


def load_split_data(csv_file, images_file):
    data = pd.read_csv(csv_file)
    images = load_images(csv_file, images_file)
    X_train = images
    y_train = data[['angle', 'speed']].values
    return X_train, y_train


# define learning rate
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 0.1
    elif epoch >15:
        lr *= 0.5
    return  lr


# model
def create_model(base_model):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)  # Assuming two outputs: angle and speed
    ])
    model.compile(optimizer = Adam(learning_rate = lr_schedule(0)), loss = 'mse', metrics=['accuracy'])
    return model


def plot_training_history(history, model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model.input_names[0]} Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model.input_names[0]} Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'{model.input_names[0]}_loss_accuracy_curves.png')  # Save the loss plot as an image file


class TrainingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"End of epoch {epoch + 1}")
        print(f"Training loss: {logs.get('loss')}, Training accuracy: {logs.get('accuracy')}")
        print(f"Validation loss: {logs.get('val_loss')}, Validation accuracy: {logs.get('val_accuracy')}")


def train_evaluate_model(df_train, df_val, image_file, base_model, target_size=(128,128), batch_size=32, epochs=30):
    model = create_model(base_model)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    monitor = TrainingMonitor()

    train_generator = create_train_generator(df_train, image_file, target_size=target_size, batch_size=batch_size)
    val_generator = create_validation_generator(df_val, image_file, target_size=target_size, batch_size=batch_size)

    early_stopper = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    history = model.fit(
        train_generator,
        steps_per_epoch = (len(df_train) // batch_size),
        validation_data = val_generator,
        # validation_steps = len(df_val) / batch_size,
        epochs = epochs,
        callbacks = [lr_scheduler, monitor, early_stopper]
    )
    
    loss, accuracy = model.evaluate(val_generator)
    print(f"Evaluation results -- Loss: {loss}, Accuracy: {accuracy}")

    return history, loss, accuracy, model



def perform_kfold_cross_validation(full_df, image_file, base_model, k=5, target_size=(128,128), batch_size=32, epochs=30):
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    best_accuracy = 0
    best_model = None
    best_fold = None

    print("Starting training process...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df), start=1):
        print(f"Fold {fold}/{k}:")
        print(f'Number of training samples: {len(train_idx)}, Number of validation samples: {len(val_idx)}')

        # Split the data into training and validation
        df_train = full_df.iloc[train_idx]
        df_val = full_df.iloc[val_idx]

        # Train and evaluate the model for the current fold
        history, loss, accuracy, model = train_evaluate_model(
            df_train, df_val, image_file, base_model,
            target_size=target_size, batch_size=batch_size, epochs=epochs
        )

        # Update the best model if the current fold's accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_fold = fold
            print(f'Updated best model at fold {fold} with accuracy: {accuracy}')

        fold_results.append({
            'fold': fold,
            'history': history,
            'loss': loss,
            'accuracy': accuracy
        })

    # Save the best model
    if best_model is not None:
        model_path = os.path.join(os.getcwd(), f'{best_model.input_names[0]}_model.h5')
        # Check if the file exists and remove it if it does
        if os.path.exists(model_path):
            os.remove(model_path)
        best_model.save(model_path)
        print(f"Training completed. Best model saved from fold {best_fold} with validation accuracy: {best_accuracy:.2f} at {model_path}")
    else:
        print("No model improvement observed across folds.")

    return fold_results


def train_multiple_models(full_df, image_file, base_models, k=5, target_size=(128,128), batch_size=32, epochs=30):
    # Returns: A dictionary with models' names as keys and their performance metrics as values.
    results = {}

    for model_creator in base_models:
        model = model_creator()
        print(f"Training model: {model.name}")

        result = perform_kfold_cross_validation(full_df, image_file, model, k = k, target_size = target_size, batch_size = batch_size, epochs = epochs)

        results[model.name] = result

    return results



def main():
    csv_file = 'training_norm.csv'
    image_file = 'training_data'
    k = 5
    target_size = (128,128)
    batch_size = 32
    epochs = 15

    full_df = load_data(csv_file)

    base_models = [
        lambda: ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
        lambda: DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
        lambda: VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    ]
    
    results = train_multiple_models(full_df, image_file, base_models, k=k, target_size=target_size, batch_size=batch_size, epochs=epochs)

    print(results)


if __name__ == '__main__':
    main()