import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Создание модели с функциональным API
inputs = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Создание модели для извлечения активаций
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Загрузка изображения и предобработка
img_path = 'test1.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_tensor = image.img_to_array(img) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

# Получение активаций
activations = activation_model.predict(img_tensor)

# Визуализация активаций без искажений
fig, axes = plt.subplots(len(activations), 1, figsize=(12, len(activations) * 2))

for ax, (layer_name, layer_activation) in zip(axes, zip([layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)], activations)):
    n_features = layer_activation.shape[-1]  # Количество фильтров
    size = layer_activation.shape[1]         # Размер feature map

    # Определяем количество строк и столбцов
    n_cols = min(8, n_features)  # Ограничиваем по горизонтали
    n_rows = int(np.ceil(n_features / n_cols))

    fig_per_layer, sub_axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    fig_per_layer.suptitle(layer_name, fontsize=12)

    sub_axes = np.array(sub_axes).reshape(-1)  # Преобразуем в 1D для удобства
    for i in range(n_features):
        channel_image = layer_activation[0, :, :, i]
        channel_image -= channel_image.mean()
        channel_image /= (channel_image.std() + 1e-5)  # Нормализация
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')

        sub_axes[i].imshow(channel_image, cmap="viridis")
        sub_axes[i].axis("off")

    for j in range(i + 1, len(sub_axes)):  # Отключаем пустые оси
        sub_axes[j].axis("off")

plt.show()
