FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Генерация данных и обучение модели при сборке (опционально)
# RUN python src/data_generator.py
# RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 



Установите зависимости:

pip install -r requirements.txt



Сгенерируйте данные и обучите модель:

python src/data_generator.py
python src/train.py


Запустите API:

uvicorn app.main:app --reload

Перейдите по адресу http://127.0.0.1:8000/docs для тестирования API через Swagger UI.



 Запуск через Docker


docker build -t churn-api .
docker run -p 8000:8000 churn-api


