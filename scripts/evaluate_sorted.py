import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Загружаем предобученную модель
model = MobileNetV2(weights="imagenet")

# Пути к отсортированным изображениям
cat_path = "data/cats_dogs/test1/cats"
dog_path = "data/cats_dogs/test1/dogs"

# Функция предсказания
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]  # Берём топ-1 предсказание
    label = decoded[1]

    return "cat" if "cat" in label else "dog"

# Проверяем точность разложенных файлов
correct = 0
total = 0

# Проверяем, правильно ли лежат коты
for filename in os.listdir(cat_path):
    file_path = os.path.join(cat_path, filename)
    if os.path.isfile(file_path):
        total += 1
        if predict(file_path) == "cat":
            correct += 1

# Проверяем, правильно ли лежат собаки
for filename in os.listdir(dog_path):
    file_path = os.path.join(dog_path, filename)
    if os.path.isfile(file_path):
        total += 1
        if predict(file_path) == "dog":
            correct += 1

# Вывод точности
accuracy = correct / total * 100
print(f"📊 Точность сортировки модели: {accuracy:.2f}% ({correct}/{total})")
