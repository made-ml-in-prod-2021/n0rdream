# Homework №1

О проекте
----------
Решается задача классификации на основе датасета https://www.kaggle.com/ronitf/heart-disease-uci  
Структура проекта выстроена, опираясь на Cookiecutter Data Science (https://drivendata.github.io/cookiecutter-data-science/#directory-structure)

Зависимости
----------    
Установку необходимых зависимостей можно осуществить, выполнив команду:
```
pip install -r requirements.txt

```

Подготовка
----------
Перед запуском пайпланов необходимо сконфигурировать yaml файлы логирования, тренировки и предсказания. Примеры таких файлов приведены в папке configs.

Обучение
----------
Пример команды на запуск обучения модели:
```
python train.py configs/training/paths.yml configs/training/preprocessing.yml configs/training/random_forest.yml
```

Предсказание
----------
Пример команды на запуск предсказания:
```
python predict.py configs/prediction/config.yml
```

Тесты
----------
Для запуска тестов выполните следующую команду:
```
python -m pytest
```
