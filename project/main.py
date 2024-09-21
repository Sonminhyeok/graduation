import os
import pandas as pd
from data_loader import load_and_prepare_data_with_user_id, cluster_user_habits, save_user_group_info, load_user_group_info
from individual_model import train_individual_model
from group_model import train_group_model
from file_watcher import watch_for_updates
num = 10
def main():
    all_data = pd.DataFrame()

    # 기존 사용자 그룹 정보를 로드
    user_group_file = './project/data/user_group_info.csv'
    prev_user_groups = load_user_group_info(user_group_file)

    # 각 파일에 사용자 ID를 추가하여 데이터 로드
    for i in range(1, num+1):
        file_path = os.path.join('./project/data', f'{i}.csv')
        data = load_and_prepare_data_with_user_id(file_path, user_id=i)  # 사용자 ID를 추가하여 데이터 로드
        all_data = pd.concat([all_data, data], ignore_index=True)

        # 개인 모델 학습 및 평가 (사용자별로 학습)
        individual_model_name = f'{i}.model'
        train_individual_model(data, individual_model_name)
    
    # 사용자별 클러스터링 진행 (기존 사용자 그룹 정보가 없을 경우 새로운 클러스터링)
    if prev_user_groups is None or '사용자 ID' not in prev_user_groups.columns or 'cluster' not in prev_user_groups.columns:
        print("이전 사용자 그룹 정보가 없거나 필드가 누락되었습니다. 새로운 클러스터링을 수행합니다.")
        user_groups = cluster_user_habits(all_data, n_clusters=3)  # 새로운 클러스터링
    else:
        user_groups = cluster_user_habits(all_data, n_clusters=3, prev_user_groups=prev_user_groups)
    
    # 사용자 그룹 정보 저장 (업데이트)
    save_user_group_info(user_groups, 'user_group_info.csv')

    # 각 클러스터별로 그룹 모델 학습
    for cluster_id in user_groups['cluster'].unique():
        all_data = all_data.reset_index(drop=True)
        user_groups = user_groups.reset_index(drop=True)

        # 이후에 코드 실행
        cluster_data = all_data[all_data['사용자 ID'].isin(user_groups[user_groups['cluster'] == cluster_id]['사용자 ID'])]
        cluster_id=int(cluster_id)
        group_model_name = f'group_{cluster_id}'  # 클러스터별로 모델 파일명을 다르게 저장
        train_group_model(cluster_data, group_model_name)

if __name__ == "__main__":
    # 폴더가 없으면 생성
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./model', exist_ok=True)
    
    # 메인 프로세스 실행
    main()
    
    # 파일 변경 감지 시작
    # watch_for_updates('./data', main)
