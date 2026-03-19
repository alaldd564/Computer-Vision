
# OpenCV, numpy, matplotlib 라이브러리 임포트
import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 계산을 위한 numpy
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib

# 이미지 파일 경로 지정
coffee_img_path = 'C:\computervision\chapter03\coffee cup.JPG'  # 사용할 이미지 경로
# 이미지 읽기 (BGR 형식)
img = cv.imread(coffee_img_path)
# 이미지가 없으면 에러 발생
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {coffee_img_path}")

# 이미지 크기(높이, 너비) 구하기
height, width = img.shape[:2]
# 사각형 너비, 높이를 이미지의 90%로 설정
rect_w = int(width * 0.9)
rect_h = int(height * 0.9)
# 사각형 좌상단 좌표 계산 (중앙 정렬)
rect_x = (width - rect_w) // 2
rect_y = (height - rect_h) // 2
# GrabCut용 사각형 (x, y, w, h)
rect = (rect_x, rect_y, rect_w, rect_h)

# GrabCut 마스크(초기값 0), 배경/전경 모델 초기화
mask = np.zeros(img.shape[:2], np.uint8)  # 마스크(0:배경, 1:전경)
bgdModel = np.zeros((1, 65), np.float64)  # GrabCut 배경 모델
fgdModel = np.zeros((1, 65), np.float64)  # GrabCut 전경 모델

# GrabCut 알고리즘 실행 (10회 반복, 사각형 기반 초기화)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

# GrabCut 결과 마스크에서 전경/가능성 전경(1), 나머지(0)로 변환
mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')
# 모폴로지 연산(열림/닫힘)으로 노이즈 제거 및 경계 보정
kernel = np.ones((5, 5), np.uint8)  # 5x5 커널
mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)  # 작은 잡음 제거
mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)  # 경계 매끄럽게

# 배경 제거된 결과 이미지 생성 (배경은 흰색)
result = img.copy()  # 원본 복사
result[mask2 == 0] = [255, 255, 255]  # 배경을 흰색으로 설정

# 결과 시각화 (원본, 마스크, 객체 추출)
plt.figure(figsize=(15, 5))  # 전체 그림 크기
plt.subplot(1, 3, 1)  # 첫 번째(원본)
plt.title('Original')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # BGR→RGB 변환
plt.axis('off')  # 축 숨김

plt.subplot(1, 3, 2)  # 두 번째(마스크)
plt.title('Mask')
plt.imshow(mask2, cmap='gray')  # 흑백 마스크
plt.axis('off')

plt.subplot(1, 3, 3)  # 세 번째(객체 추출)
plt.title('Object Extracted')
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))  # BGR→RGB 변환
plt.axis('off')

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 출력
