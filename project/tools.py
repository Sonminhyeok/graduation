import re
import pandas as pd
import torch
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

def add_kobert_embeddings(data, tokenizer, model):
    embeddings = []
    for menu_text in data['breakfast']:
        menu_text = str(menu_text)  # Ensure menu_text is a string
        tokens = tokenizer(menu_text, return_tensors='pt')
        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
        # 임베딩 차원 축소를 위해 PCA 적용
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 임베딩을 데이터프레임의 새로운 열로 추가
    data['breakfast_kobert'] = list(reduced_embeddings)
    # 임베딩을 데이터프레임의 새로운 열로 추가
    
    return data
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
df = pd.read_csv("./project/data/cleaned_main_data1000.csv",encoding="cp949")
df = add_kobert_embeddings(df, tokenizer, model)
df.to_csv("./project/data/cleaned_main_data10002.csv",encoding="cp949",index=False)