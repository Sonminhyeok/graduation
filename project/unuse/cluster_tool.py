import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
import os
def load_user_group_info(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding='cp949')
    return None



def clustering_function():
    # 파`일 불러오기 (data1.csv ~ data10.csv)
    file_list = glob.glob('./project/data/main_data*.csv')

    # 사용자의 데이터를 담을 리스트
    user_data = []

    # 각 파일에서 데이터를 읽어와 처리
    for file in file_list:
        df = pd.read_csv(file, encoding='cp949')
        
        # 조식, 중식, 석식 임베딩을 모두 평탄화하여 하나의 벡터로 결합
        df['breakfast_flat'] = df['breakfast_embedding'].apply(lambda x: eval(x)[:3] + eval(x)[3])
        df['lunch_flat'] = df['lunch_embedding'].apply(lambda x: eval(x)[:3] + eval(x)[3])
        df['dinner_flat'] = df['dinner_embedding'].apply(lambda x: eval(x)[:3] + eval(x)[3])
        # 조식, 중식, 석식 임베딩 벡터를 하나의 벡터로 결합
        df['combined_embedding'] = df.apply(lambda row: row['breakfast_flat'] + row['lunch_flat'] + row['dinner_flat'], axis=1)
        
        # 사용자의 데이터를 평균 벡터로 통합
        user_vector = np.mean(df['combined_embedding'].tolist(), axis=0)
        
        user_data.append(user_vector)


    # NumPy 배열로 변환
    X = np.array(user_data)

    # 스케일링 (정규화)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    prev_user_groups = load_user_group_info('./project/data/user_group_info.csv')

        # KMeans 클러스터링 및 이전 클러스터 번호 유지
    kmeans = KMeans(n_clusters=3, random_state=42)
    if prev_user_groups is not None:
        prev_clusters = prev_user_groups.set_index('사용자 ID')['cluster']
        
        # 사용자가 이전 클러스터 정보가 있는지 체크
        cluster_result = []
        for i, file in enumerate(file_list):
            file_name = file.split("\\")[1].split(".")[0].split("data")[1]
            # 이전 클러스터 정보가 있는 경우 해당 클러스터를 유지
            if int(file_name) in prev_clusters.index:
                cluster = prev_clusters[int(file_name)]
            else:
                # 새로운 클러스터를 할당
                cluster = kmeans.fit_predict(X_scaled[i].reshape(1, -1))[0]

            cluster_result.append({
                '사용자 ID': file_name,
                'cluster': cluster
            })
    else:
        # 이전 데이터가 없으면 새로운 클러스터를 할당
        labels = kmeans.fit_predict(X_scaled)
        cluster_result = []
        for i, file in enumerate(file_list):
            file_name = file.split("\\")[1].split(".")[0].split("data")[1]
            cluster_result.append({
                '사용자 ID': file_name,
                'cluster': labels[i]
            })

    # 결과 저장
    df_result = pd.DataFrame(cluster_result)
    df_result.to_csv("./project/data/user_group_info.csv", index=False, encoding='cp949')
if __name__ == "__main__":
    clustering_function()