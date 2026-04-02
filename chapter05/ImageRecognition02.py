"""
CIFAR-10 데이터셋을 활용한 CNN 이미지 분류기
"""
import tensorflow as tf  # 텐서플로우 임포트
from tensorflow.keras import layers, models  # 케라스 레이어, 모델 임포트
from tensorflow.keras.datasets import cifar10  # CIFAR-10 데이터셋 임포트
import numpy as np  # 넘파이 임포트
import matplotlib.pyplot as plt  # 그래프 시각화

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # CIFAR-10 데이터셋 불러오기
x_train = x_train.astype('float32') / 255.0  # 훈련 이미지 정규화
x_test = x_test.astype('float32') / 255.0    # 테스트 이미지 정규화

# 2. CNN 모델 구성 (성능 개선)
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),  # 합성곱층1
    layers.BatchNormalization(),  # 배치 정규화
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),  # 합성곱층2
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),  # 풀링층1
    layers.Dropout(0.25),        # 드롭아웃1

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),  # 합성곱층3
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),  # 합성곱층4
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),  # 풀링층2
    layers.Dropout(0.25),        # 드롭아웃2

    layers.Conv2D(128, (3,3), padding='same', activation='relu'), # 합성곱층5
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), padding='same', activation='relu'), # 합성곱층6
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),  # 풀링층3
    layers.Dropout(0.25),        # 드롭아웃3

    layers.Flatten(),  # 1차원 변환
    layers.Dense(256, activation='relu'),  # 완전연결층
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # 드롭아웃4
    layers.Dense(10, activation='softmax')  # 출력층(10개 클래스)
])

# 3. 컴파일 및 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 모델 컴파일
# epoch 30으로 증가
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)  # 모델 훈련

# 4. 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)  # 테스트 데이터 평가
print(f"테스트 정확도: {acc*100:.2f}%")  # 정확도 출력

# 5. 학습 곡선 시각화
plt.figure(figsize=(10,6))  # 그림 크기 키우기
plt.plot(history.history['accuracy'], label='Train Acc')  # 학습 정확도 곡선
plt.plot(history.history['val_accuracy'], label='Val Acc')  # 검증 정확도 곡선
plt.xlabel('Epoch')  # x축: 에폭
plt.ylabel('Accuracy')  # y축: 정확도
plt.legend()  # 범례
plt.title('CIFAR-10 CNN learning curve')  # 그래프 제목
plt.show()  # 그래프 출력

# 6. 테스트 이미지(dog.jpg) 예측 (파일이 있을 경우)
import os  # OS 라이브러리
from tensorflow.keras.preprocessing import image  # 이미지 전처리

img_path = 'chapter05/dog.jpg'  # 예측할 이미지 경로
if os.path.exists(img_path):  # 파일 존재 여부 확인
    img = image.load_img(img_path, target_size=(32,32))  # 이미지 로드 및 크기 변환
    img_array = image.img_to_array(img) / 255.0  # 배열 변환 및 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    pred = model.predict(img_array)  # 예측
    class_idx = np.argmax(pred)  # 예측 클래스 인덱스
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']  # 클래스명
    print(f"dog.jpg 예측 결과: {class_names[class_idx]}")  # 예측 결과 출력
else:
    print("dog.jpg 파일이 chapter05 폴더에 없습니다.")  # 파일 없을 때 메시지
