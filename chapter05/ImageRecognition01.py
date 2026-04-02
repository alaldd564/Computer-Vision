"""
간단한 이미지 분류기 구현 (MNIST)
"""
import tensorflow as tf  # 텐서플로우 라이브러리 임포트
from tensorflow.keras import layers, models  # 케라스의 레이어, 모델 임포트
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋 임포트
import matplotlib.pyplot as plt  # 그래프 시각화 라이브러리

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터셋 불러오기
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0  # 훈련 이미지 1차원화 및 정규화
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0    # 테스트 이미지 1차원화 및 정규화

# 2. 모델 구성
model = models.Sequential([  # 순차적 신경망 모델 생성
    layers.Input(shape=(28*28,)),  # 입력층: 784차원(28x28)
    layers.Dense(128, activation='relu'),  # 은닉층1: 128개 노드, ReLU 활성화
    layers.Dense(64, activation='relu'),   # 은닉층2: 64개 노드, ReLU 활성화
    layers.Dense(10, activation='softmax') # 출력층: 10개(0~9), 소프트맥스
])

# 3. 컴파일 및 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 모델 컴파일(최적화/손실/평가지표)
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)  # 모델 훈련(5에폭, 미니배치 32, 검증 10%)

# 4. 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)  # 테스트 데이터로 평가
print(f"테스트 정확도: {acc*100:.2f}%")  # 정확도 출력

# 5. 학습 곡선 시각화
plt.figure(figsize=(10,6))  # 그림 크기 키우기
plt.plot(history.history['accuracy'], label='Train Acc')  # 학습 정확도 곡선
plt.plot(history.history['val_accuracy'], label='Val Acc')  # 검증 정확도 곡선
plt.xlabel('Epoch')  # x축: 에폭
plt.ylabel('Accuracy')  # y축: 정확도
plt.legend()  # 범례 표시
plt.title('MNIST 분류기 학습 곡선')  # 그래프 제목
plt.show()  # 그래프 출력
