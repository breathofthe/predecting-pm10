# PM10 예측 모델

## 프로젝트 개요
이 프로젝트는 PM10 농도를 예측하기 위해 CNN+LSTM 모델을 사용하여 다양한 기상 및 오염 변수와의 관계를 분석하고 모델을 학습합니다.

## 데이터 처리
1. **결측치 처리**: 선형보간법 사용
2. **이상치 제거**: IQR 방식을 이용하여 이상치 제거
3. **정규화**: StandardScaler를 사용하여 데이터를 정규화

## 상관분석
NO2와 O3가 PM10과 가장 큰 상관관계를 보였습니다.

## 데이터 전처리
- 결측값 확인 및 처리 (평균 대체법 사용)
- 이상치 제거 후 데이터 크기: 7682
- 정규화된 데이터 사용

## 모델 학습
### CNN+LSTM 모델 사용
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_cnn_lstm_model(filters=64, kernel_size=3, neurons=50, learning_rate=0.001):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(window_size, X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        LSTM(neurons, return_sequences=False),
        Dense(neurons, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
    return model
```

## 하이퍼파라미터 튜닝
 GridSearchCV를 사용하여 최적의 파라미터를 찾았습니다
 ```python
 param_grid = {
    'model__filters': [32, 64],
    'model__kernel_size': [2, 3],
    'model__neurons': [50],
    'model__learning_rate': [0.001],
    'batch_size': [16]
}
```
## 성능 평가
- `MAE`: 0.5087
- `MSE`: 0.4870
- `RMSE`: 0.6978

## 결론
`CNN+LSTM` 모델을 사용하여 `PM10` 농도를 예측할 수 있으며, 모델의 성능은 `MAE` `0.5087`, `RMSE` `0.6978`로 나타났습니다

