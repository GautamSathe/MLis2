import numpy as np
import re
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img


def create_test_generator(test_images, target_size = (128, 128)):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_images,
        target_size=target_size,
        batch_size=1,
        shuffle=False,
        class_mode=None 
    )
    return test_generator



def predict_images(model_path, test_images, target_size = (128, 128)):
    test_generator = create_test_generator(test_images, target_size)
    model = load_model(model_path)
    file_ids = []

    prediction = model.predict(test_generator, steps = len(test_generator))

    filenames = test_generator.filenames
    for filename in filenames:
        match = re.search(r'\d+', filename)
        if match:
            file_ids.append(match.group())

    return file_ids, prediction


def final_predict(models_path, test_images, target_size = (128, 128)):
    predictions = []

    for model_path in models_path:
        file_id, prediction = predict_images(model_path, test_images, target_size)
        predictions.append(prediction)
        print(model_path)
        print(prediction)
    predictions = np.mean(predictions, axis=0)
    result = list(zip(file_id, predictions))
    return result

    
def trans_to_csv(results, save_path = 'predictions.csv'):
    data = {
        'image_id': [r[0] for r in results],
        'angle': [r[1][0] for r in results],
        'speed': [r[1][1] for r in results]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def main():
    test_images = 'Data/test_data'
    target_size = (128, 128)
    models_path = ['Retrain_model/densenet121_input_model']
    results = final_predict(models_path, test_images, target_size)
    trans_to_csv(results)
    

if __name__ == '__main__':
    main()