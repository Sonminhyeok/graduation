import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from individual_model import process_date, process_time_of_day
from sklearn.preprocessing import StandardScaler

# 사용자별로 음식을 임베딩하여 클러스터링을 위한 데이터 준비


# 데이터를 로드하고 사용자별 데이터를 준비하는 함수 (사용자 ID 추가)
def load_and_prepare_data_with_user_id(file_path, user_id):
    # CSV 파일을 불러오고 사용자 ID를 추가
    data = pd.read_csv(file_path)
    data['사용자 ID'] = user_id  # 사용자 ID 추가


    return data

# 사용자별 클러스터링 함수
def cluster_user_habits(data, n_clusters=3, prev_user_groups=None):
    if '사용자 ID' not in data.columns:
        raise KeyError("데이터에 '사용자 ID' 필드가 없습니다. 사용자 ID를 추가하세요.")
    
    # 날짜 및 식사시간을 클러스터링을 위한 숫자로 변환
    data = process_date(data)  # 연, 월, 일로 변환
    data = process_time_of_day(data)  # 아침, 점심, 저녁 숫자로 변환
   
    # 사용자 ID 기준으로 그룹화하여 사용자별 데이터를 준비 (embedding, 날짜, 식사시간으로 그룹화)
    user_groups = data.select_dtypes(include='number').groupby('사용자 ID').mean()  # 사용자별 평균을 계산
    print(user_groups)
    
    # 특성 스케일링
    scaler = StandardScaler()
    features_to_cluster = ['embedding', '연도', '월', '일', '식사시간_encoded']
    user_groups_scaled = scaler.fit_transform(user_groups[features_to_cluster])
    
    # 이전 클러스터링 데이터가 있다면 클러스터 번호를 유지
    if prev_user_groups is not None:
        if '사용자 ID' not in prev_user_groups.columns or 'cluster' not in prev_user_groups.columns:
            raise KeyError("'사용자 ID' 또는 'cluster' 필드가 이전 사용자 그룹 정보에 없습니다.")
        user_groups = pd.merge(user_groups, prev_user_groups[['사용자 ID', 'cluster']], on='사용자 ID', how='left')

    # KMeans 클러스터링을 수행하여 cluster 열을 추가
    if 'cluster' not in user_groups.columns:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_groups['cluster'] = kmeans.fit_predict(user_groups_scaled)

    # 새로운 사용자 (cluster 정보가 없는 경우만) 처리
    new_users = user_groups[user_groups['cluster'].isnull()]
    if not new_users.empty:
        new_users_scaled = scaler.transform(new_users[features_to_cluster])
        new_users['cluster'] = kmeans.fit_predict(new_users_scaled)
        user_groups.update(new_users)

    return user_groups


# 이전에 저장된 사용자 그룹 정보를 로드하는 함수
def load_user_group_info(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# 사용자 그룹 정보를 CSV로 저장하는 함수
def save_user_group_info(user_groups, group_file_name):
    group_file_path = os.path.join('./project/data', group_file_name)
    
    print(user_groups)
    print(user_groups.columns)  # 저장 전 열 이름 확인

    user_groups = user_groups.reset_index()

    user_groups.to_csv(group_file_path, index=False)
    print(f"사용자 그룹 정보가 {group_file_path}에 저장되었습니다.")
