import pandas as pd
import ast
import numpy as np

# Load the dataset
file_path = './project/data/main_data1000.csv'
df = pd.read_csv(file_path, encoding='cp949')

# Identify embedding columns
embedding_columns = [col for col in df.columns if 'embedding' in col]

# Convert strings to lists
for col in embedding_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x)[:3] if isinstance(x, str) else x)

# Remove outliers from embedding columns using the Interquartile Range (IQR) method
def remove_embedding_outliers(df, embedding_columns, threshold=3):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in embedding_columns:
        # Convert embedding values to numpy arrays
        df[column] = df[column].apply(lambda x: np.array(x) if isinstance(x, list) else x)
        # Stack embeddings for statistical calculations
        embeddings = np.stack(df[column].values)
        # Calculate mean and standard deviation for each component in the embedding
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        # Define the mask to filter out rows with any component of the embedding beyond the threshold
        mask = np.all(np.abs((embeddings - mean) / std) < threshold, axis=1)
        df = df[mask]
    return df

# Remove outliers from the dataset
cleaned_data = remove_embedding_outliers(df, embedding_columns)

# Replace [0, 0, 0] with the mean vector for each embedding column
for col in embedding_columns:
    # Calculate the mean vector excluding [0, 0, 0]
    valid_vectors = [vec for vec in cleaned_data[col] if isinstance(vec, np.ndarray) and not np.all(vec == [0, 0, 0])]
    if valid_vectors:
        means = np.mean(valid_vectors, axis=0).tolist()
    else:
        means = [0, 0, 0]  # Set to a default value if no valid vectors exist
    # Replace [0, 0, 0] with the calculated mean
    cleaned_data[col] = cleaned_data[col].apply(lambda x: means if isinstance(x, np.ndarray) and np.all(x == [0, 0, 0]) else x)

# Format embeddings as properly formatted strings with commas between numbers
def format_embeddings_as_strings(df, embedding_columns):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in embedding_columns:
        df.loc[:, column] = df[column].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x)
    return df

# Format the embedding columns to ensure proper comma-separated formatting
cleaned_data = format_embeddings_as_strings(cleaned_data, embedding_columns)

# Save the cleaned dataset
cleaned_file_path = './project/data/cleaned_main_data1000.csv'
cleaned_data.to_csv(cleaned_file_path, index=False, encoding='cp949')

print(f"Cleaned dataset saved as '{cleaned_file_path}'.")
