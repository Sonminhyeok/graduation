import re
import pandas as pd
import torch

def plot_y_test_vs_y_pred_line(y_test, y_pred):
    import matplotlib.pyplot as plt
    # 데이터의 길이 확보
    n_samples = len(y_test)

    # 각 차원에 대해 선 그래프 그리기
    for i in range(y_test.shape[1]):
        plt.figure(figsize=(14, 6))
        plt.plot(range(n_samples), y_test[:, i], label='Actual (y_test)', linestyle='-', color='b')
        plt.plot(range(n_samples), y_pred[:, i], label='Predicted (y_pred)', linestyle='--', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel(f'Value for Dimension {i+1}')
        plt.title(f'Comparison of Actual vs Predicted for Dimension {i+1}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
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
    embeddings = []
    for menu_text in data['lunch']:
        menu_text = str(menu_text)  # Ensure menu_text is a string
        tokens = tokenizer(menu_text, return_tensors='pt')
        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
        # 임베딩 차원 축소를 위해 PCA 적용
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 임베딩을 데이터프레임의 새로운 열로 추가
    data['lunch_kobert'] = list(reduced_embeddings)
    embeddings = []
    for menu_text in data['dinner']:
        menu_text = str(menu_text)  # Ensure menu_text is a string
        tokens = tokenizer(menu_text, return_tensors='pt')
        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
        # 임베딩 차원 축소를 위해 PCA 적용
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 임베딩을 데이터프레임의 새로운 열로 추가
    data['dinner_kobert'] = list(reduced_embeddings)
    
    return data
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import glob
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def main():

    file_list = glob.glob('./project/data/cleaned_main_data*.csv')
 
    for file in file_list:

        df = pd.read_csv(file,encoding="cp949")
        df = add_kobert_embeddings(df, tokenizer, model)
        df.to_csv(file,encoding="cp949",index=False)
        
if __name__ == "__main__":
    main()