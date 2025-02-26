# 🐱🐶 (pawlytics?)

Простая модель, написанная в рамках домашнего задания. Определяет, кто на фото — **кошка или собака**.

## 🚀 Установка и запуск

### 1️⃣ Установите зависимости
```bash
pip install -r requirements.txt
```

### 2️⃣ Запустите классификацию изображения
```bash
python scripts/predict.py --image путь/к/фото.jpg
```

## 📂 Структура проекта
```
📂 models/          # Сохранённые модели
📂 data/            # Датасеты
📂 scripts/         # Исходный код
📂 plots/           # Графики обучения
📂 test_images/     # Фото для тестирования модели
```

## 🛠 Описание скриптов
| Скрипт                  | Описание |
|-------------------------|----------|
| `train.py`             | PROD Обучает модель на датасете `cats_dogs` |
| `sort_images.py`       | PROD Сортирует тестовые изображения по папкам `cats/` и `dogs/` с помощью обученной модели |
| `predict.py`           | PROD Делает предсказание класса (`кошка` или `собака`) для одного изображения |
| `evaluate.py`          | TEST Оценивает точность модели на тестовом датасете |
| `evaluate_sorted.py`   | TEST Проверяет точность сортировки изображений, сделанной `sort_images.py` |
| `split_to_layers.py`   | TEST Разбирает модель на слои (используется для проверки архитектуры) |
