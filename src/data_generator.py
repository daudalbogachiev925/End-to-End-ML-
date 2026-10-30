import pandas as pd
import numpy as np
import os

def generate_data(n_samples=1000):
    np.random.seed(42)
    
    # Генерация признаков
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Tenure': np.random.randint(1, 72, n_samples), # Месяцев с нами
        'MonthlyCharges': np.random.uniform(20.0, 120.0, n_samples),
        'TotalCharges': lambda x: x['Tenure'] * x['MonthlyCharges'],
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
    }
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges']
    
    # Создаем целевую переменную (Churn: 1 - ушел, 0 - остался)
    # Логика: если контракт месячный и цена высокая - вероятность ухода выше
    churn_prob = 0.1 
    churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.4, 0)
    churn_prob += np.where(df['MonthlyCharges'] > 80, 0.2, 0)
    churn_prob -= np.where(df['Tenure'] > 24, 0.2, 0)
    
    df['Churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1))
    
    # Сохранение
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/churn.csv', index=False)
    print("Датасет успешно сгенерирован и сохранен в data/churn.csv")

if __name__ == "__main__":
    generate_data()
