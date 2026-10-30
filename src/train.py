import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    # 1. Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv('data/churn.csv')
    
    # 2. Разделение на X и y
    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn']
    
    # 3. Определение типов колонок
    numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'InternetService']
    
    # 4. Создание пайплайна предобработки
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 5. Создание полного пайплайна (предобработка + модель)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # 6. Разделение на train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Обучение
    print("Обучение модели...")
    model.fit(X_train, y_train)
    
    # 8. Оценка
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 9. Сохранение модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    print("Модель сохранена в models/churn_model.pkl")

if __name__ == "__main__":
    train_model()
