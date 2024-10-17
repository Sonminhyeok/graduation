import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import glob
import os
from tools import process_date

def evaluate_model_performance(y_test, y_pred):
    # R² Score 계산 (결정 계수)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae

def create_time_series_features(data, n_days=3, meal='breakfast'):
    features = []
    targets = []
    
    embedding_column = f'{meal}_embedding'
    cuisine_column = f'{meal}_cuisine'
    kobert_column = f'{meal}_kobert'
    count_column = f'{meal}_count'
    
    for i in range(n_days, len(data)):
        current_features = data.loc[i, ['연도', '월', '일', count_column, '요일', cuisine_column, kobert_column]].values
        past_embeddings = []
        for j in range(1, n_days + 1):
            embedding = eval(data.loc[i - j, embedding_column])[:3]
            past_embeddings.extend(embedding)
        features.append(np.concatenate((current_features, past_embeddings)))
        target = eval(data.loc[i, embedding_column])[:3]
        targets.append(target)

    return np.array(features), np.array(targets)

def train_model_with_extended_features(data, n_days, meal):
    X, y = create_time_series_features(data, n_days=n_days, meal=meal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
   
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 1.0, 0.5, 0.6],
        'colsample_bytree': [0.8, 1.0],
    }
    
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_scaled)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, y_train_scaled)
    
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    plot_y_test_vs_y_pred_line(y_test, y_pred)
    return evaluate_model_performance(y_test, y_pred)

def plot_y_test_vs_y_pred_line(y_test, y_pred):
    import matplotlib.pyplot as plt
    # 데이터의 길이 확보
    n_samples = len(y_test)

    # 각 차원에 대해 선 그래프 그리기
    for i in range(y_test.shape[1]):
        plt.figure(figsize=(14, 6))
        plt.plot(range(n_samples), y_test[:, i], label='Actual (y_test)', linestyle='-', color='b')
        plt.plot(range(n_samples), y_pred[:, i], label='Predicted (y_pred)', linestyle='--', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel(f'Value for Dimension {i+1}')
        plt.title(f'Comparison of Actual vs Predicted for Dimension {i+1}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
def main():
    file_list = ['./project/data/cleaned_main_data2291.csv']
    n_days_list = [2]
    
    for file in file_list:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, encoding='cp949')
        process_date(df)
        df['요일'] = df['날짜'].dt.dayofweek
        
        le = LabelEncoder()
        for meal in ['breakfast', 'lunch', 'dinner']:
            df[f'{meal}_cuisine'] = le.fit_transform(df[f'{meal}_cuisine'])
            df[f'{meal}_kobert'] = le.fit_transform(df[f'{meal}_kobert'])
            df[f'{meal}_count'] = df.get(f'{meal}_count', 0)
        
        results = []
        for n_days in n_days_list:
            for meal in ['breakfast', 'lunch', 'dinner']:
                print(f"Training model for {meal} with n_days = {n_days}...")
                r2, mae = train_model_with_extended_features(df, n_days, meal)
                results.append({'Meal': meal, 'n_days': n_days, 'R2 Score': r2, 'MAE': mae})
        
        # DataFrame으로 변환
        results_df = pd.DataFrame(results)
        
        # 파일명에서 경로와 확장자 제거 후 결과 파일 생성
        file_name = os.path.basename(file).replace('.csv', '')
        output_file_path = f'./project/data/result/{file_name}_performance_results.csv'
        
        # 결과 저장
        # results_df.to_csv(output_file_path, index=False)
        # print(f'Results saved to {output_file_path}')

if __name__ == "__main__":
    main()
