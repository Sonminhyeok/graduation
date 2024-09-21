import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 날짜 변환 함수 (연, 월, 일로 분리)
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


# 개인 모델 학습 및 평가 함수
def train_individual_model(data, model_name):
    # 날짜 데이터를 연, 월, 일로 분리
    data = process_date(data)
    
    # 시간대(아침, 점심, 저녁)를 숫자로 변환
    data = process_time_of_day(data)
    
    # X는 날짜(연도, 월, 일) + 시간대, y는 메뉴 임베딩
    X = data[['연도', '월', '일', '식사시간_encoded']]  # 날짜 및 시간대 데이터를 사용
    data = data.astype({'embedding':'int'})
   
    y = data['embedding']                         # 예측할 대상은 메뉴
    
    # 학습 데이터를 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 랜덤 포레스트 모델 사용
    model = RandomForestClassifier()
    
    # 모델이 이미 존재하면 로드
    model_path = os.path.join('./project/model', model_name)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 학습 데이터 평가
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"{model_name} 훈련 데이터 정확도: ", train_accuracy)
    
    # 테스트 데이터 평가
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{model_name} 테스트 데이터 정확도: ", test_accuracy)
    
    # 모델 저장
    joblib.dump(model, model_path)
    
    return model

