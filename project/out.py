# Convert columns to numeric where possible and handle non-numeric data for outlier detection
# Select the numeric columns for outlier analysis
import numpy as np
numeric_columns = ['breakfast_count', 'lunch_count', 'dinner_count']

# Define a function to remove outliers using the Interquartile Range (IQR) method
def remove_embedding_outliers(df, embedding_columns, threshold=3):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in embedding_columns:
        # Convert the string representation of lists to actual lists of floats
        df.loc[:, column] = df[column].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
        # Calculate the mean and standard deviation for each component in the embedding
        embeddings = np.stack(df[column].values)
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        # Define the mask to filter out rows with any component of the embedding beyond the threshold
        mask = np.all(np.abs((embeddings - mean) / std) < threshold, axis=1)
        df = df[mask]
    return df
def format_embeddings_as_strings(df, embedding_columns):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in embedding_columns:
        # Format embeddings as properly formatted strings with commas between numbers
        df.loc[:, column] = df[column].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x)
    return df

# Format the embedding columns to ensure proper comma-separated formatting

# Specify the embedding columns to process
embedding_columns = ['breakfast_embedding', 'lunch_embedding', 'dinner_embedding']

# Remove outliers from embedding columns using the z-score threshold

# Remove outliers from the dataset
import pandas as pd
data = pd.read_csv("./project/data/main_data1000.csv",encoding='cp949')

cleaned_data = remove_embedding_outliers(data, embedding_columns)
cleaned_data = format_embeddings_as_strings(cleaned_data, embedding_columns)
cleaned_data.to_csv("./project/data/cleaned_main_data1000.csv",encoding='cp949',index=False)