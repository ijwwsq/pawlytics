import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Загружаем модель
model = tf.keras.models.load_model("models/best_model.h5")

# Папки с правильными картинками
base_path = "/home/egor/code/ml/data/cats_dogs"
cat_folder = os.path.join(base_path, "cats")
dog_folder = os.path.join(base_path, "dogs")

correct = 0
total = 0
sample_size = 200  # Количество картинок для теста

print("---")

def check_images(folder, true_label):
    global correct, total
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    random.shuffle(files)  # Перемешиваем файлы
    files = files[:sample_size]  # Берём только 1000 картинок

    for filename in files:
        img_path = os.path.join(folder, filename)

        # Загружаем и обрабатываем картинку
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание
        prediction = model.predict(img_array, verbose=0)[0][0]
        predicted_label = "dog" if prediction > 0.5 else "cat"

        # Проверяем, правильно ли угадала модель
        if predicted_label == true_label:
            correct += 1
        
        total += 1

# Проверяем обе папки (по 1000 картинок)
check_images(cat_folder, "cat")
check_images(dog_folder, "dog")

# Выводим точность
accuracy = (correct / total) * 100
print(f"📊 Точность модели на 2000 картинках: {accuracy:.2f}% ({correct}/{total})")
