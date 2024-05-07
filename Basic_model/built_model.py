import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.applications import VGG19, DenseNet121, ResNet50
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import img_to_array, load_img
from data_generator import create_train_generator, create_validation_generator


def load_data(csv_path):
    return pd.read_csv(csv_path)
    #dataframe


def load_images_and_labels(csv_file, images_file, target_size):
    images = []
    labels = []
    csv_data = load_data(csv_file)

    for idx, row in csv_data.iterrows():
        img_path = os.pathpjoin(images_file, row['image_id'])
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize the image data to 0-1
        images.append(img_array)
        labels.append([row['angle'], row['speed']])

    np_images = np.array(images)
    np_labels = np.array(labels)
    return np_images, np_labels
    


# define learning rate
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 0.1
    elif epoch >15:
        lr *= 0.5
    return  lr


class TrainingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"End of epoch {epoch + 1}")
        print(f"Training loss: {logs.get('loss')}, Training accuracy: {logs.get('accuracy')}")
        print(f"Validation loss: {logs.get('val_loss')}, Validation accuracy: {logs.get('val_accuracy')}")


def setup_callbacks():
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min', restore_best_weights=True)
    monitor = TrainingMonitor()
    return [lr_scheduler, early_stopper, monitor]


def train_evaluate_model(df_train, df_val, image_file, model, target_size=(128,128), batch_size=32, epochs=30):
    callbacks = setup_callbacks()
    steps_per_epoch = (len(df_train) // batch_size)
    validation_steps = int(np.ceil(len(df_val) / batch_size))

    train_generator = create_train_generator(df_train, image_file, target_size=target_size, batch_size=batch_size)
    val_generator = create_validation_generator(df_val, image_file, target_size=target_size, batch_size=batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        validation_data = val_generator,
        validation_steps = validation_steps,
        workers=10,             # Number of parallel worker processes
        use_multiprocessing = True,
        epochs = epochs,
        callbacks = callbacks
    )

    loss, accuracy = model.evaluate(val_generator)
    print(f"Evaluation results -- Loss: {loss}, Accuracy: {accuracy}")

    return history, loss, accuracy, model



def perform_kfold_cross_validation(full_df, image_file, base_model, model_name, k=5, target_size=(128,128), batch_size=32, epochs=30):
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
        model_path = os.path.join(os.getcwd(), f'{model_name}_model')
        # Check if the file exists and remove it if it does
        if os.path.exists(model_path):
            os.remove(model_path)
        best_model.save(model_path)
        print(f"Training completed. Best model saved from fold {best_fold} with validation accuracy: {best_accuracy:.2f} at {model_path}")
    else:
        print("No model improvement observed across folds.")

    return fold_results


def create_model(base_model):
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)  # Assuming two outputs: angle and speed
    ])
    
    model.compile(optimizer = Adam(learning_rate = lr_schedule(0)), loss = 'mse', metrics=['accuracy'])
    return model


def train_multiple_models(full_df, image_file, base_models, k=5, target_size=(128,128), batch_size=32, epochs=30):
    # Returns: A dictionary with models' names as keys and their performance metrics as values.
    results = {}
    
    for base_model in base_models:
        base_model = base_model()
        model = create_model(base_model)
        model_name = base_model.name
        print(f"Training model: {model_name}")

        result = perform_kfold_cross_validation(full_df, image_file, model, model_name, k = k, target_size = target_size, batch_size = batch_size, epochs = epochs)

        results[model_name] = result

    return results



def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    csv_file = '/home/alyzf6/MLis2/training_norm.csv'
    image_file = '/home/alyzf6/MLis2/Data/training_data'
    k = 5
    target_size = (128,128)
    batch_size = 32
    epochs = 15

    full_df = load_data(csv_file)

    base_models = [
        lambda: VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
        lambda: ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
        lambda: DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    ]
    
    results = train_multiple_models(full_df, image_file, base_models, k=k, target_size=target_size, batch_size=batch_size, epochs=epochs)

    print(results)


if __name__ == '__main__':
    main()