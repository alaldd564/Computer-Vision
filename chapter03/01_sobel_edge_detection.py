
# OpenCV, numpy, matplotlib 라이브러리 임포트
import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 계산용 numpy
import matplotlib.pyplot as plt  # 시각화용 matplotlib

# 이미지 파일 경로 지정
gray_img_path = 'C:\computervision\chapter03\edgeDetectionImage.jpg'  # 사용할 이미지 경로
# 이미지 읽기 (BGR 형식)
img = cv.imread(gray_img_path)
# 이미지가 없으면 에러 발생
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {gray_img_path}")

# 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 소벨 필터로 x축 방향 에지 검출
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
# 소벨 필터로 y축 방향 에지 검출
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# x, y 방향 에지로부터 에지 강도 계산
magnitude = cv.magnitude(sobelx, sobely)

# 에지 강도 이미지를 uint8로 변환
edge_img = cv.convertScaleAbs(magnitude)

# 결과 시각화 (원본, 에지 강도)
plt.figure(figsize=(10, 5))  # 전체 그림 크기
plt.subplot(1, 2, 1)  # 첫 번째(원본)
plt.title('Original')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # BGR→RGB 변환
plt.axis('off')  # 축 숨김

plt.subplot(1, 2, 2)  # 두 번째(에지 강도)
plt.title('Sobel Edge Magnitude')
plt.imshow(edge_img, cmap='gray')  # 흑백으로 표시
plt.axis('off')

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 출력
