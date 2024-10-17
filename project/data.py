import pandas as pd
from gpt_func import analyze_food_with_gpt, analyze_food_with_crawling
from cluster_tool import clustering_function
from tools import remove_brackets
import glob
import numpy as np
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, encoding='cp949')
    df['조식']=df['조식'].apply(remove_brackets)
    df['중식']=df['중식'].apply(remove_brackets)
    df['석식']=df['석식'].apply(remove_brackets)
    df.sort_values(by='날짜')
    df.to_csv(file_path, encoding='cp949', index=False)
    return df

def drop_function(data):
    
    data['조식열량'] = data['조식열량'].str.replace('kcal', '').astype(float)
    data['중식열량'] = data['중식열량'].str.replace('kcal', '').astype(float)
    data['석식열량'] = data['석식열량'].str.replace('kcal', '').astype(float)

    # 조식, 중식, 석식 칼로리를 합산하여 '총열량'이라는 새 열을 추가
    data['총열량'] = data[['조식열량', '중식열량', '석식열량']].sum(axis=1)

    # 각 날짜별로 최대 칼로리를 가진 행만 남기기 위해 idxmax() 사용
    # '날짜' 열을 기준으로 그룹화하고, '총열량'이 가장 높은 행을 선택
    # filtered_data = data.loc[data.groupby('날짜')['총열량'].idxmax()]

    # 필요한 열만 남기기 (예: 날짜, 조식, 중식, 석식)
    filtered_data = data[['날짜', '조식', '중식', '석식']]

    # 결과를 확인
    return filtered_data


def embed_data_gpt(data):
    data['breakfast_embedding'] = data['조식'].apply(analyze_food_with_gpt)
    data['lunch_embedding'] = data['중식'].apply(analyze_food_with_gpt)
    data['dinner_embedding'] = data['석식'].apply(analyze_food_with_gpt)
    return data


def embed_data_crawling(data):
    data['breakfast_embedding'] = data['조식'].apply(analyze_food_with_crawling)
    data['lunch_embedding'] = data['중식'].apply(analyze_food_with_crawling)
    data['dinner_embedding'] = data['석식'].apply(analyze_food_with_crawling)
    return data




def fillna_func(df):
    # NaN 값을 채우기 위한 기본 embedding 값
    default_embedding = [0, 0, 0, [1, 0, 0, 0, 0]]
    # 데이터프레임의 embedding 열에 대해 NaN 값을 기본값으로 채움
    df['breakfast_embedding'] = df['breakfast_embedding'].apply(lambda x: default_embedding if pd.isna(x) else x)
    df['lunch_embedding'] = df['lunch_embedding'].apply(lambda x: default_embedding if pd.isna(x) else x)
    df['dinner_embedding'] = df['dinner_embedding'].apply(lambda x: default_embedding if pd.isna(x) else x)
    return df
def fillna_func_mean(df):
    import numpy as np
    # 각 embedding 열의 평균값 계산
    # embedding이 리스트 형태로 저장된 경우 리스트의 각 요소의 평균을 계산
    def calculate_mean_embedding(column_name):
        # 각 embedding을 평가하고 NaN이 아닌 값들만 필터링
        embeddings = df[column_name].dropna().apply(lambda x: eval(x)[:3]).tolist()  # 앞의 3개만 사용
        
        # 각 위치의 평균 계산 (앞의 3개 요소에 대해 평균을 계산)
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding

    # 각 embedding 열의 평균값 계산
    mean_breakfast_embedding = calculate_mean_embedding('breakfast_embedding')
    mean_lunch_embedding = calculate_mean_embedding('lunch_embedding')
    mean_dinner_embedding = calculate_mean_embedding('dinner_embedding')

    # NaN 값을 각 열의 평균 embedding 값으로 채움
    df['breakfast_embedding'] = df['breakfast_embedding'].apply(
        lambda x: mean_breakfast_embedding if pd.isna(x) else x
    )
    df['lunch_embedding'] = df['lunch_embedding'].apply(
        lambda x: mean_lunch_embedding if pd.isna(x) else x
    )
    df['dinner_embedding'] = df['dinner_embedding'].apply(
        lambda x: mean_dinner_embedding if pd.isna(x) else x
    )
    
    return df
#메뉴 등장 횟수 추가
def count_menu(df):
    df['breakfast_count'] = df['breakfast'].map(df['breakfast'].value_counts())
    df['lunch_count'] = df['lunch'].map(df['lunch'].value_counts())
    df['dinner_count'] = df['dinner'].map(df['dinner'].value_counts())
    return df

def extract_main_foods(df):
    results = []

    # 날짜별 그룹화
    grouped = df.groupby('날짜')

    for date, group in grouped:
        # 각 식사별 총 영양 성분 합계 계산 및 조건에 따라 메인 음식 선택
        def select_main_food(group, embedding_column, menu_column):
            # 각 메뉴의 embedding 합계 계산, 리스트가 아닌 경우에는 0으로 처리
            group['embedding_sum'] = group[embedding_column].apply(
                lambda x: sum([v for v in x[:3] if isinstance(v, (int, float))]) if isinstance(x, list) else 0
            )
            
            # 합계가 가장 큰 메뉴의 인덱스를 찾음
            max_idx = group['embedding_sum'].idxmax()
            main_food = group.loc[max_idx, menu_column]
            main_food_embedding = group.loc[max_idx, embedding_column]
            
            # main_food가 문자열인지 확인하고 특정 조건을 만족하는지 확인
            if isinstance(main_food, str) and ('우유' in main_food or '밥' in main_food or (main_food.endswith('김치') and len(main_food) == len('김치') + 2)):
                # 합계가 두 번째로 큰 메뉴를 선택
                remaining_group = group.drop(max_idx)
                
                # 남은 그룹이 비어 있지 않은지 확인
                if not remaining_group.empty:
                    second_max_idx = remaining_group['embedding_sum'].idxmax()
                    main_food = group.loc[second_max_idx, menu_column]
                    main_food_embedding = group.loc[second_max_idx, embedding_column]
                else:
                    # 남은 그룹이 비어 있다면 기본값 반환 (None 또는 다른 처리)
                    return None, None

            return main_food, main_food_embedding

        # 각 식사에 대해 메인 음식 선택 및 해당 embedding 값 추출
        breakfast_main, breakfast_embedding = select_main_food(group, 'breakfast_embedding', '조식')
        lunch_main, lunch_embedding = select_main_food(group, 'lunch_embedding', '중식')
        dinner_main, dinner_embedding = select_main_food(group, 'dinner_embedding', '석식')

        # 결과 저장
        results.append({
            '날짜': date,
            'breakfast': breakfast_main,
            'breakfast_embedding': breakfast_embedding,
            'lunch': lunch_main,
            'lunch_embedding': lunch_embedding,
            'dinner': dinner_main,
            'dinner_embedding': dinner_embedding
        })

    # 결과를 데이터프레임으로 변환
    return pd.DataFrame(results)


def extract_embedding_sums(df):
    # NaN 및 기본값 처리
    results = []

    # 날짜별 그룹화
    grouped = df.groupby('날짜')

    for date, group in grouped:
        # 각 식사별 embedding 합계 계산
        def calculate_embedding_sum(group, embedding_column):
            # embedding 값들의 합을 계산
            total_embedding_sum = sum([sum(v[:3]) for v in group[embedding_column]])
            return total_embedding_sum

        # 각 식사에 대해 embedding 합계 계산
        breakfast_sum = calculate_embedding_sum(group, 'breakfast_embedding')
        lunch_sum = calculate_embedding_sum(group, 'lunch_embedding')
        dinner_sum = calculate_embedding_sum(group, 'dinner_embedding')

        # 결과 저장
        results.append({
            '날짜': date,
            'breakfast_embedding': breakfast_sum,
            'lunch_embedding': lunch_sum,
            'dinner_embedding': dinner_sum
        })

    # 결과를 데이터프레임으로 변환
    return pd.DataFrame(results)





def main():

    file_list = glob.glob('./project/data/embedded_data*.csv')
    # fillna [000] and maindish ver
    
    
    for file in file_list:
        df= load_and_prepare_data(file)
        df = fillna_func(df)
        
        df = extract_main_foods(df)
 
        
        df = count_menu(df)
       
        file_name=file.split("\\")[1].split(".")[0].split("data")[1]
        df.to_csv(f'./project/data/main_data{file_name}.csv', encoding='cp949', index=False)
    
    
    # 각 파일에서 데이터를 읽어와 크롤링
    
    # file_list = glob.glob('./project/data/before/data*.csv')
    # for file in file_list:
    #     print(file)
    #     df = load_and_prepare_data(file)
    #     # df = drop_function(df)
    #     df = embed_data_crawling(df)
    #     file_name=file.split("\\")[1].split(".")[0]
    #     df.to_csv(f'./project/data/before/embedded_{file_name}.csv', encoding='cp949', index=False)
        
        
if __name__ == "__main__":
    
    main()