# Homework №2

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
Сервис написан на фреймворке FastAPI. 
Сервис запускается посредством docker-контейнера.  
Использовались следующие команды:
* Создание образа
```
docker build -t nordream/inference_service:v1 .
```
* Загрузка образа на dockerhub:
```
docker push nordream/inference_service:v1
```
* Скачивание образа:
```
docker pull nordream/inference_service:v1
```
* Запуск сервиса в контейнере:
```
docker run -p 8000:8000 nordream/inference_service:v1
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
