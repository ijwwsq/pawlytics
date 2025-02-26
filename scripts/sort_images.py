import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Это скрипт, который берет тестовые неразмеченные 
# данные и с помощью обученной модели распределяет 
# их по двум папкам

# Загружаем модель
model = tf.keras.models.load_model("models/best_model.h5")

# Пути
test_path = "data/cats_dogs/test1"
cat_path = os.path.join(test_path, "cats")
dog_path = os.path.join(test_path, "dogs")

# Создаём папки для сортировки
os.makedirs(cat_path, exist_ok=True)
os.makedirs(dog_path, exist_ok=True)

# Функция предсказания
def predict_and_sort(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    is_dog = prediction > 0.5
    target_folder = dog_path if is_dog else cat_path

    shutil.move(img_path, os.path.join(target_folder, os.path.basename(img_path)))

# Запускаем предсказание и сортировку
for filename in os.listdir(test_path):
    file_path = os.path.join(test_path, filename)
    if os.path.isfile(file_path):
        predict_and_sort(file_path)

print("Сортировка завершена!")
