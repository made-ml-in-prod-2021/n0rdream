# Homework №1

О проекте
----------
Онлайн-сервис по выдаче предсказаний в задаче классификации на основе датасета https://www.kaggle.com/ronitf/heart-disease-uci

Зависимости
----------    
Установку необходимых зависимостей можно осуществить, выполнив команду:
```
pip install -r requirements.txt
```

О сервисе
----------
Сервис запускается внутри docker-контейнера при помощи следующих команд:
```
docker build -t inference_service .
docker run -p 8000:8000 inference_service
```
Также образ был загружен на dockerhub при помощи команды:
```
docker push nordream/inference_service:v1
```
Скачать его можно командой:
```
docker pull nordream/inference_service:v1
```

Получение предсказаний
----------
Необходимые файлы находятся в директории prediction_request.
Настройте конфиг config_request.yaml.
Запустить скрипт можно следующей командой:
```
python make_request.py config_request.yaml
```

Тестирование
----------
Для запуска тестов выполните следующую команду:
```
python -m pytest tests/
```
