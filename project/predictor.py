import os
import pandas as pd
from tensorflow.keras.models import load_model

# 예측을 위한 데이터 전처리 함수 (날짜 및 시간대 처리)
def prepare_input_data(input_date, input_time):
    # 날짜 전처리 (연도, 월, 일로 분리)
    input_date = pd.to_datetime(input_date)  # 날짜 입력을 datetime으로 변환
    year = input_date.year
    month = input_date.month
    day = input_date.day
    
    # 시간대 전처리 (아침: 0, 점심: 1, 저녁: 2)
    time_mapping = {'아침': 0, '점심': 1, '저녁': 2}
    time_encoded = time_mapping.get(input_time, -1)  # 예외 처리: 유효하지 않은 시간 입력
    
    if time_encoded == -1:
        raise ValueError("잘못된 시간대 입력입니다. 아침, 점심, 저녁 중 하나를 선택하세요.")
    
    # 전처리된 데이터를 DataFrame으로 반환
    input_data = pd.DataFrame([[year, month, day, time_encoded]], columns=['연도', '월', '일', '식사시간_encoded'])
    return input_data

# 사용자 클러스터 예측 및 그룹 모델을 사용해 embedding 값 예측
def predict_embedding(input_date, input_time, user_id, user_groups):
    # 입력 데이터 전처리
    input_data = prepare_input_data(input_date, input_time)
    
    # 사용자 ID 기반으로 클러스터 확인
    user_cluster = user_groups[user_groups['사용자 ID'] == user_id]['cluster'].values
    if len(user_cluster) == 0:
        raise ValueError("해당 사용자의 클러스터 정보를 찾을 수 없습니다.")
    
    cluster_id = int(user_cluster[0])  # 사용자의 클러스터 ID

    # 그룹 모델 로드 (해당 클러스터에 맞는 모델)
    group_model_name = f'group_{cluster_id}.h5'
    group_model_path = f'./project/model/{group_model_name}'
    print(group_model_path)
    try:
        group_model = load_model(group_model_path)  # 모델 로드
        print(f"모델이 정상적으로 로드되었습니다: {group_model_name}")
    except Exception as e:
        raise RuntimeError(f"모델을 로드하는 중 오류가 발생했습니다: {str(e)}")
    
    # 그룹 모델을 사용해 embedding 값 예측
    predicted_embedding = group_model.predict(input_data)
    
    return predicted_embedding

# 예시: 특정 날짜와 시간으로 예측 수행
def main_predict():
    user_groups = pd.read_csv('./project/data/user_group_info.csv')  # 사용자 그룹 정보 로드
    
    # 날짜와 시간 입력 (예시로 '2024-09-21', '점심'을 사용)
    input_date = '2024-09-21'
    input_time = '점심'
    user_id = 1  # 예시 사용자 ID
    
    try:
        # embedding 값 예측
        predicted_embedding = predict_embedding(input_date, input_time, user_id, user_groups)
        print(f"Predicted embedding for user {user_id} on {input_date} {input_time}: {predicted_embedding}")
    except ValueError as e:
        print(f"입력 오류: {e}")
    except RuntimeError as e:
        print(f"모델 로드 오류: {e}")

if __name__ == "__main__":
    main_predict()
