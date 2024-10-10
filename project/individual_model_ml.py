import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from tools import process_date
from sklearn.model_selection import GridSearchCV

def evaluate_model_performance(y_test, y_pred):
    # R² Score 계산 (결정 계수)
    r2 = r2_score(y_test, y_pred)
    r2_percentage = r2 * 100
    print(f'R² Score: {r2_percentage:.2f}%')

    # MAPE 계산 (평균 절대 백분율 오차)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    print(f'MAPE: {mape:.2f}%')
    
    


# 아침/점심/저녁 정보를 숫자로 변환하는 함수
def process_time_of_day(data):
    # 시간대에 따른 숫자 변환 (아침: 0, 점심: 1, 저녁: 2)
    time_of_day_mapping = {'아침': 0, '점심': 1, '저녁': 2}
    data['식사시간_encoded'] = data['식사시간'].map(time_of_day_mapping)
    return data



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train_model_with_date_embeddings(data):
    process_date(data)
    # 날짜 데이터와 음식 임베딩 데이터를 나누기
    X = data[['연도', '월', '일','breakfast_count']].values  # 날짜 정보를 연도, 월, 일로 변환한 값들 
    #input은 최근 1주일간의 식단정보. output은 그다음날의 식단정보로 수정
    
    y = np.stack(data['breakfast_embedding'].apply(lambda x: eval(x)[:3]))  # 다차원 음식 임베딩 벡터
    
    # 트레인 셋과 테스트 셋을 9:1 비율로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 데이터 스케일링
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    #param grid
    param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.0001,0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 1.0, 0.5, 0.6],
    'colsample_bytree': [0.8, 1.0],
    
    }
    # # MLP Regressor 모델 학습
    # model = MLPRegressor(hidden_layer_sizes=(64, 128, 64), activation='relu', max_iter=50000, random_state=42)
    
    # # randomforest regressor 모델 학습
    # model = RandomForestRegressor()
    
    #xgboost
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_scaled)

    # 최적의 하이퍼파라미터 출력
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best R² Score: {grid_search.best_score_}")
    
    model.fit(X_train_scaled, y_train_scaled)
    # 예측 및 성능 평가
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    print(y_pred)
    # 평균 제곱 오차 계산
    mse = mean_squared_error(y_test, y_pred)
    # print(f'Mean Squared Error: {mse}')
    # print(evaluate_model_performance(y_test, y_pred))
    return model, scaler_x, scaler_y

def main():
    df = pd.read_csv("./project/data/sum_data1000.csv", encoding = 'cp949')
    model, scaler_x, scaler_y=train_model_with_date_embeddings(df)
    # print(model)
 
    
if __name__ == "__main__":
    
    main()
