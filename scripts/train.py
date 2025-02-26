import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Пути к файлам и папкам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATASET_PATH = os.path.join(BASE_DIR, '../data/cats_dogs')  
MODEL_PATH = os.path.join(BASE_DIR, '../models/best_model.h5')  
PLOT_PATH = os.path.join(BASE_DIR, '../plots/training_progress.png')  

# Конфигурация данных
IMG_SIZE = (128, 128)  # Размер изображений
BATCH_SIZE = 64        # Размер батча
EPOCHS = 10            # Количество эпох

# Аугментация данных (добавляем случайные изменения для улучшения обучения)
datagen = ImageDataGenerator(
    rescale=1./255,           # Нормализация пикселей
    rotation_range=20,        # Повороты до 20°
    width_shift_range=0.2,    # Горизонтальные сдвиги
    height_shift_range=0.2,   # Вертикальные сдвиги
    shear_range=0.2,          # Сдвиг углов
    zoom_range=0.2,           # Приближение/отдаление
    horizontal_flip=True,     # Отражение по горизонтали
    fill_mode='nearest'       # Заполнение пикселей после трансформаций
)

# Загружаем данные
train_data = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# Создаем сверточную нейросеть
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),  # Сжатие карты признаков

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),                     # Разворачиваем вектор признаков
    layers.Dense(256, activation='relu'),  
    layers.Dropout(0.5),                  # Отключаем случайные нейроны для предотвращения переобучения
    layers.Dense(1, activation='sigmoid') # Выходной слой (0 - кошка, 1 - собака)
])

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
              loss='binary_crossentropy', metrics=['accuracy'])

# Запуск обучения
history = model.fit(train_data, epochs=EPOCHS)

# Сохраняем график обучения
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Точность')
plt.plot(history.history['loss'], label='Потери')
plt.legend()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH)
plt.show()

# Сохранение модели
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f'✅ Модель сохранена: {MODEL_PATH}')
