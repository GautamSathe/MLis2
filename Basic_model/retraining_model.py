import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback, EarlyStopping
from data_generator import create_train_generator, create_validation_generator



# define learning rate
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 0.1
    elif epoch >15:
        lr *= 0.5
    return  lr


def create_model(model):
    for layer in model.layers[:-4]:
        layer.trainable = False
    model.compile(optimizer = Adam(learning_rate = lr_schedule(0)), loss = 'mse', metrics=['accuracy'])
    return model
    

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


def train_evaluate_model(df_train, df_val, image_file, model, train_data_num, target_size=(128,128), batch_size=32, epochs=30):
    callbacks = setup_callbacks()
    steps_per_epoch = (train_data_num // batch_size)
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



def perform_kfold_cross_validation(full_df, image_file, base_model, model_name, train_data_num, k=5, target_size=(128,128), batch_size=32, epochs=30):
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
            df_train, df_val, image_file, base_model, train_data_num,
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


def train_models(full_df, image_file, model_paths, train_data_num, k, target_size, batch_size, epochs):
    results = {}
 
    for model_path in model_paths:
        model = load_model(model_path)
        model_name = model.input_names[0]
        model = create_model(model)
        print(f"Training model: {model_name}")

        result = perform_kfold_cross_validation(full_df, image_file, model, model_name, train_data_num, k = k, target_size = target_size, batch_size = batch_size, epochs = epochs)

        results[model_name] = result

    return results

def trans_to_tlite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


def main():
    csv_file = 'new_images_norm.csv'
    image_file = 'new_images'
    k = 5
    target_size = (128,128)
    batch_size = 32
    epochs = 15
    train_data_num = 4000

    full_df = pd.read_csv(csv_file)

    model_paths = [
        '/home/alyzf6/MLis2/Generate_models/densenet121_model',
        '/home/alyzf6/MLis2/Generate_models/resnet50_model',
        '/home/alyzf6/MLis2/Generate_models/vgg19_model'
        ]
    
    results = train_models(full_df, image_file, model_paths, train_data_num, k=k, target_size=target_size, batch_size=batch_size, epochs=epochs)

    print(results)

def main1():
    csv_file = 'new_images_norm.csv'
    image_file = 'new_images'
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        if not os.path.isfile(os.path.join(image_file, row['image_id'])):
            # print(os.path.isfile(os.path.join(image_file, str(row['image_id']))))
            print(f"Invalid or missing file: {row['image_id']} at index {index}")

def main2():
    model_path = 'Retrain_model/densenet121_input_model'
    trans_to_tlite(model_path)


if __name__ == '__main__':
    main2()