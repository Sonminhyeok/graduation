import re
import pandas as pd
def remove_brackets(menu):
    if not isinstance(menu, str):
        return ""
    return re.sub(r'\([^)]*\)', '', menu).strip()


# 날짜 변환 함수 (연, 월, 일로 분리)
def process_date(data):
    # 괄호 내 텍스트 (요일 등) 제거
    data['날짜'] = data['날짜'].str.replace(r"\(.*\)", "", regex=True).str.strip()
    # 날짜 데이터를 datetime 형식으로 변환
    data['날짜'] = pd.to_datetime(data['날짜'], format='%Y-%m-%d', errors='coerce')
    # 연, 월, 일 컬럼 추가
    data['연도'] = data['날짜'].dt.year
    data['월'] = data['날짜'].dt.month
    data['일'] = data['날짜'].dt.day
    return data