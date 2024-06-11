

pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 불러오기
file_path = '/content/PM10_fin_2.csv'
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# 훈련 데이터와 테스트 데이터로 분할
train_data = data['2020-09-01':'2021-09-30']
test_data = data['2021-10-01':'2021-12-31']

# 데이터 스케일링
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.iloc[:, 1:])
test_scaled = scaler.transform(test_data.iloc[:, 1:])

def create_dataset(data, time_step_X=4, time_step_Y=3):
    X, Y = [], []
    for i in range(len(data) - max(time_step_X, time_step_Y)):
        X.append(data[i:(i + time_step_X)])
        Y.append(data[i + time_step_Y, 3])  # PM10은 네 번째 열 (인덱스 3)
    return np.array(X), np.array(Y)

# 데이터셋 생성
time_step_Y = 72  # 3일간의 데이터
time_step_X = 96  # 4일간의 데이터
X_train, y_train = create_dataset(train_scaled, time_step_X, time_step_Y)
X_test, y_test = create_dataset(test_scaled, time_step_X, time_step_Y)

# CNN 입력 형식으로 데이터 reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# CNN 모델 생성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step_X, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))

# 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 성능 평가
mae = mean_absolute_error(y_test, test_predict)
mse = mean_squared_error(y_test, test_predict)
rmse = np.sqrt(mse)

print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')
