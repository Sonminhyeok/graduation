import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from xgboost import XGBRegressor
from tools import process_date
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from tools import process_date
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def evaluate_model_performance(y_test, y_pred):
    # R² Score 계산 (결정 계수)
    r2 = r2_score(y_test, y_pred)
    r2_percentage = r2
    print(f'R² Score: {r2_percentage:.2f}')
    mae = mean_absolute_error(y_test,y_pred)
    print(f'MAE: {mae:.2f}')
    


def create_time_series_features(data, n_days=3):
    # 날짜 정보와 이전 n일간의 임베딩 데이터를 포함하는 새로운 feature를 추가한 데이터프레임 생성
    features = []
    targets = []
    
    for i in range(n_days, len(data)):
        # X: 현재 날짜 정보 (연도, 월, 일, breakfast_count, breakfast_cuisine, 요일) + 이전 n일간의 임베딩 데이터
        # current_features = data.loc[i, [ '월', '요일', 'breakfast_cuisine']].values
        current_features = data.loc[i, ['연도', '월', '일', 'breakfast_count', '요일', 'breakfast_cuisine','breakfast_kobert']].values
        # 이전 n일간의 embedding 데이터 (각각이 다차원 베터)
        past_embeddings = []
        for j in range(1, n_days + 1):
            embedding = eval(data.loc[i - j, 'breakfast_embedding'])[:3]  # 이전 j일의 임베딩 데이터
            past_embeddings.extend(embedding)  # 베터를 하나로 확장
            
        # 현재 날짜 정보 + 이전 n일간의 임베딩 데이터를 합쳐 feature로 사용
        features.append(np.concatenate((current_features, past_embeddings)))
        
        # y: 현재 날짜의 임베딩 (다차원 베터)
        target = eval(data.loc[i, 'breakfast_embedding'])[:3]
        targets.append(target)

    return np.array(features), np.array(targets)

def train_model_with_extended_features(data, n_days):
    # 3일 전부터의 데이터를 이용하여 새로운 feature set 생성
    X, y = create_time_series_features(data, n_days=n_days)
    
    # 트레인 셋과 테스트 셋을 9:1 비율로 나름기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
   
    # 데이터 스케일링
    # scaler_x = RobustScaler()
    # scaler_y = RobustScaler()
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # XGBoost 모델 학습
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

    # 최적의 하이퍼파리메터 출력
    print(f"Best Parameters: {grid_search.best_params_}")
    
    
    # 최적의 모델로 학습
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, y_train_scaled)
    
    # 예측 및 성능 평가
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    evaluate_model_performance(y_test, y_pred)
    return best_model, scaler_x, scaler_y, y_test, y_pred

def plot_y_test_vs_y_pred_line(y_test, y_pred):
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
    df = pd.read_csv("./project/data/cleaned_main_data10002.csv", encoding='cp949')
    
    process_date(df)
    df['요일'] = df['날짜'].dt.dayofweek
    
    # Add encoding for breakfast_cuisine
    le = LabelEncoder()
    df['breakfast_cuisine'] = le.fit_transform(df['breakfast_cuisine'])
    df['breakfast_kobert'] = le.fit_transform(df['breakfast_kobert'])
    start = [2,14]
    for i in start:
        model, scaler_x, scaler_y, y_test, y_pred = train_model_with_extended_features(df, i)
        plot_y_test_vs_y_pred_line(y_test, y_pred)

if __name__ == "__main__":
    main()