import random
import pandas as pd
import os
from datetime import timedelta, datetime
embedding_set = pd.read_csv("./project/data/food_embeddings_manual.csv")
# 한국식 대표 메뉴 리스트
menu_list = [
    "김치찌개", "된장찌개", "순두부찌개", "비빔밥", "불고기", "잡채", "갈비탕", "냉면", "떡볶이",
    "삼겹살", "제육볶음", "김밥", "라면", "설렁탕", "칼국수", "해물파전", "삼계탕", "물냉면", "비빔냉면",
    "부대찌개", "콩나물국밥", "동태찌개", "돼지갈비", "치킨", "족발", "보쌈", "닭갈비", "순대국",
    "감자탕", "육개장", "초밥", "돈까스", "우동", "짜장면", "짬뽕", "회", "낙지볶음", "굴국밥",
    "매운탕", "양념게장", "간장게장", "볶음밥", "닭볶음탕", "치즈돈까스", "카레", "수제비", "쌀국수",
    "어묵탕", "김치볶음밥", "순대볶음", "고등어구이", "오징어볶음", "파전", "된장국", "차돌박이",
    "육회", "회덮밥", "샤브샤브", "떡국", "콩국수", "라멘", "참치김밥", "갈비찜", "김치전", "북엇국",
    "전복죽", "오리구이", "낙곱새", "코다리조림", "치즈떡볶이", "돼지국밥", "알밥", "명란파스타",
    "깐풍기", "마파두부", "스팸김치찌개", "계란말이", "닭가슴살샐러드", "돈까스카레", "새우튀김",
    "차슈덮밥", "양장피", "샐러드파스타", "우삼겹덮밥", "훈제오리샐러드", "불닭볶음면", "매운치킨",
    "함박스테이크", "크림파스타", "해물탕", "스테이크", "스시롤", "아구찜", "고기국수", "냉채족발",
    "돈까스덮밥", "치즈라면", "새우볶음밥", "갈릭버터새우", "쭈꾸미볶음", "로제떡볶이", "양꼬치",
    "곱창전골", "떡만둣국", "새우장", "고추장불고기", "샌드위치", "만두국", "비빔국수", "닭강정",
    "유부초밥", "돼지고기김치찜", "바지락칼국수", "훈제연어샐러드", "잔치국수", "닭한마리", "간장새우"
]

# 데이터 생성 함수
def generate_diet_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    data = []

    for date in date_range:
        # 하루에 아침, 점심, 저녁 3번의 식사 데이터를 생성
        for meal_time in ['아침', '점심', '저녁']:
            menu = random.choice(menu_list)
            calories = random.randint(300, 900)  # 칼로리는 300~900 사이 랜덤
            data.append([date, meal_time, menu, calories])

    # DataFrame으로 변환
    df = pd.DataFrame(data, columns=['날짜', '식사시간', '메뉴', '칼로리'])
    
    return df

# CSV 저장 함수
def add_embeddings_to_diet_data(diet_df, embedding_df):
    # 메뉴를 기준으로 diet_df와 embedding_df를 병합 (left join 방식)
    merged_df = pd.merge(diet_df, embedding_df, how='left', left_on='메뉴', right_on='food')
    merged_df['embedding']*=100
    # 필요 없는 'menu' 열 제거
    merged_df = merged_df.drop(columns=['food'])

    return merged_df

# 5. CSV 저장 함수
def save_to_csv(df, file_name,i):
    os.makedirs('./data', exist_ok=True)
    file_path = os.path.join('./project/data', file_name)
    df['user_id']=i
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"CSV 파일이 {file_path}에 저장되었습니다.")

# 6. 기간 설정 예시
start_date = '2023-09-01'
end_date = '2024-09-30'

# 7. 데이터 생성
diet_data = generate_diet_data(start_date, end_date)

# 8. embedding 추가
diet_data_with_embeddings = add_embeddings_to_diet_data(diet_data, embedding_set)

# 9. 생성된 데이터를 CSV 파일로 저장
num =10
for i in range(1,num+1):
    save_to_csv(diet_data_with_embeddings, f'{i}.csv',i)

