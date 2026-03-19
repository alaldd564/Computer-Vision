
# OpenCV, numpy, matplotlib 라이브러리 임포트
import cv2 as cv  # OpenCV 라이브러리
import numpy as np  # 수치 계산용 numpy
import matplotlib.pyplot as plt  # 시각화용 matplotlib

# 이미지 파일 경로 지정
dabo_img_path = 'C:\computervision\chapter03\dabo.jpg'  # 사용할 이미지 경로
# 이미지 읽기 (BGR 형식)
img = cv.imread(dabo_img_path)
# 이미지가 없으면 에러 발생
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {dabo_img_path}")

# 캐니 에지 검출 (threshold1=100, threshold2=200)
edges = cv.Canny(img, 100, 200)

# 허프 변환을 이용한 직선 검출 (더 많은 직선 검출을 위해 파라미터 조정)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=60, maxLineGap=20)

# 원본 이미지 복사 (직선 표시용)
img_lines = img.copy()
# 검출된 직선이 있으면 반복하며 그림
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 직선의 양 끝점 좌표
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색, 두께 2

# 결과 시각화 (원본, 직선 검출 결과만)
plt.figure(figsize=(10, 5))  # 전체 그림 크기
plt.subplot(1, 2, 1)  # 첫 번째(원본)
plt.title('Original')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # BGR→RGB 변환
plt.axis('off')  # 축 숨김

plt.subplot(1, 2, 2)  # 두 번째(직선 검출 결과)
plt.title('Detected Lines')
plt.imshow(cv.cvtColor(img_lines, cv.COLOR_BGR2RGB))  # BGR→RGB 변환
plt.axis('off')

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 출력
