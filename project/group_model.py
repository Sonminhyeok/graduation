import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

def process_date(data):
    data['날짜'] = pd.to_datetime(data['날짜'])  # 날짜 데이터를 datetime 형식으로 변환
    data['연도'] = data['날짜'].dt.year  # 연도
    data['월'] = data['날짜'].dt.month  # 월
    data['일'] = data['날짜'].dt.day    # 일
    return data

# 아침/점심/저녁 정보를 숫자로 변환하는 함수
def process_time_of_day(data):
    # 시간대에 따른 숫자 변환 (아침: 0, 점심: 1, 저녁: 2)
    time_of_day_mapping = {'아침': 0, '점심': 1, '저녁': 2}
    data['식사시간_encoded'] = data['식사시간'].map(time_of_day_mapping)
    return data
# 그룹 모델 학습 함수
def train_group_model(data, group_model_name):
    
    X = data[['연도', '월', '일', '식사시간_encoded']]  # 날짜 및 시간대 데이터를 사용
    data = data.astype({'embedding':'int'})
    
    y = data['embedding']  
    
    
    model = Sequential([
        Dense(64, input_dim=X.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 학습 데이터를 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 그룹 모델이 이미 존재하면 로드
    group_model_path = os.path.join('./project/model', group_model_name)
    if os.path.exists(group_model_path):
        model = load_model(group_model_path)
    
    # 모델 학습 (Validation data를 사용하여 검증)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # 모델 저장 (.h5 확장자를 사용하여 저장)
    model.save(group_model_path + '.h5')
    
    # # 모델 평가
    evaluation_results = model.evaluate(X_test, y_test)
    print(f"모델 평가 결과 (테스트 데이터): 손실(Loss): {evaluation_results[0]}, MAE: {evaluation_results[1]}")
    
    
    
    return model, history
