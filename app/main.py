from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Инициализация приложения
app = FastAPI(title="Churn Prediction API", description="API для предсказания оттока клиентов")

# Загрузка модели при старте
try:
    model = joblib.load('models/churn_model.pkl')
except Exception as e:
    model = None
    print(f"Ошибка загрузки модели: {e}. Убедитесь, что модель обучена.")

# Схема входных данных (Pydantic)
class CustomerData(BaseModel):
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str

# Схема ответа
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    message: str

@app.get("/")
def read_root():
    return {"message": "API для предсказания оттока работает. Перейдите на /docs для тестирования."}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    # Преобразование входных данных в DataFrame (как при обучении)
    input_df = pd.DataFrame([{
        'Tenure': data.Tenure,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges,
        'Contract': data.Contract,
        'InternetService': data.InternetService
    }])
    
    # Предсказание
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Вероятность класса 1 (ушел)
    
    message = "Клиент, скорее всего, уйдет" if prediction == 1 else "Клиент останется"
    
    return PredictionResponse(
        churn_prediction=int(prediction),
        churn_probability=float(probability),
        message=message
    )
