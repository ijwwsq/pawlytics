import tensorflow as tf
import numpy as np
import argparse
import os
from tensorflow.keras.preprocessing import image

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, '../models/best_model.h5')  

IMG_SIZE = (128, 128)  # Размер изображений

# Загружаем обученную модель
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    """Предсказывает, кто на фото: кошка или собака"""
    img = image.load_img(image_path, target_size=IMG_SIZE)  # Загружаем изображение
    img_array = image.img_to_array(img) / 255.0  # Нормализуем
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность батча

    prediction = model.predict(img_array)[0][0]  # Получаем предсказание
    predicted_class = 'Собака' if prediction > 0.5 else 'Кошка'  # Определяем класс
    confidence = max(prediction, 1 - prediction) * 100  # Уверенность в предсказании

    print(f'📷 Изображение: {image_path}')
    print(f'🐾 Предсказание: {predicted_class} ({confidence:.2f}%)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Классификация изображения (кошка или собака)")
    parser.add_argument("image_path", type=str, help="Путь к изображению")
    args = parser.parse_args()

    predict_image(args.image_path)
