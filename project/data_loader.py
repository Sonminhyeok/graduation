import pandas as pd
import ast
import numpy as np
import glob

def remove_embedding_outliers(df, embedding_columns, threshold=3):
    df = df.copy()  
    for column in embedding_columns:
        df[column] = df[column].apply(lambda x: np.array(x) if isinstance(x, list) else x)
        embeddings = np.stack(df[column].values)
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        mask = np.all(np.abs((embeddings - mean) / std) < threshold, axis=1)
        df = df[mask]
    return df

def format_embeddings_as_strings(df, embedding_columns):
    df = df.copy()  
    for column in embedding_columns:
        df.loc[:, column] = df[column].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x)
    return df
def col_mean_func(df):
    embedding_columns = [col for col in df.columns if 'embedding' in col]

    for col in embedding_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x)[:3] if isinstance(x, str) else x)
        
    
    
    cleaned_data = remove_embedding_outliers(df, embedding_columns)

    for col in embedding_columns:
        valid_vectors = [vec for vec in cleaned_data[col] if isinstance(vec, np.ndarray) and not np.all(vec == [0, 0, 0])]
        if valid_vectors:
            means = np.mean(valid_vectors, axis=0).tolist()
        else:
            means = [0, 0, 0]  
        cleaned_data[col] = cleaned_data[col].apply(lambda x: means if isinstance(x, np.ndarray) and np.all(x == [0, 0, 0]) else x)

    cleaned_data = format_embeddings_as_strings(cleaned_data, embedding_columns)
    return cleaned_data

def main():
    file_list = glob.glob('./project/data/main_data*.csv')
    for file in file_list:
        print(file)
        df = pd.read_csv(file, encoding='cp949')
        cleaned_data = col_mean_func(df)
        file_name=file.split("\\")[1]
        cleaned_data.to_csv(f"./project/data/cleaned_{file_name}", index=False, encoding='cp949')

if __name__=='__main__':
    main()