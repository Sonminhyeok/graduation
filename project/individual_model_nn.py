import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tools import process_date
def build_mlp_model(input_dim, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def build_lstm_model(input_dim, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(input_dim, 1), return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def train_model_with_date_embeddings(data):
    process_date(data)
    # 날짜 데이터와 음식 임베딩 데이터를 나누기
    X = data[['연도', '월', '일','breakfast_count']].values  # 날짜 정보를 연도, 월, 일로 변환한 값들
    y = np.stack(data['breakfast_embedding'].apply(lambda x: eval(x)[:3]))  # 다차원 음식 임베딩 벡터 (길이 3)
    
    # 트레인 셋과 테스트 셋을 9:1 비율로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 데이터 스케일링
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # MLP 모델 정의 및 학습
    input_dim = X_train_scaled.shape[1]
    output_dim = y_train_scaled.shape[1]
    model = build_lstm_model(input_dim, output_dim)

    # 모델 학습 (에포크와 배치 크기는 조정 가능)
    model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    # 예측 및 성능 평가
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 평균 제곱 오차 및 R² 점수 계산
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) * 100
    
    
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2:.2f}%')


    return model, scaler_x, scaler_y

def main():
    df = pd.read_csv("./project/data/main_data1000.csv", encoding='cp949')
    model, scaler_x, scaler_y = train_model_with_date_embeddings(df)

if __name__ == "__main__":
    main()