import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # numpy 임포트
import os  # OS 경로 처리를 위한 os 임포트

# 이미지 파일 경로 설정
img_path = os.path.join(os.path.dirname(__file__), 'images/rose.png')  # 이미지 경로 생성
img = cv2.imread(img_path)  # 이미지 읽기

if img is None:
    raise FileNotFoundError('이미지 파일을 찾을 수 없습니다.')  # 파일 없을 때 예외 처리

# 이미지 크기를 400x400으로 고정
fixed_size = (400, 400)  # 고정 크기 설정
img_resized = cv2.resize(img, fixed_size)  # 이미지 리사이즈

# 이미지 중심 계산
(h, w) = fixed_size  # 높이, 너비
center = (w // 2, h // 2)  # 중심 좌표

# 회전 + 크기 조절
angle = 30  # +30도 회전
scale = 0.8  # 크기 0.8로 조절
M = cv2.getRotationMatrix2D(center, angle, scale)  # 회전 행렬 생성

# 평행이동 적용 (x축 +80, y축 -40)
M[0, 2] += 80  # x축 이동
M[1, 2] += -40  # y축 이동

# 변환 적용
rotated_scaled_translated = cv2.warpAffine(img_resized, M, fixed_size)  # 변환 적용

# 결과 시각화
cv2.imshow('Original', img_resized)  # 원본 이미지 출력
cv2.imshow('Rotated + Scaled + Translated', rotated_scaled_translated)  # 변환 이미지 출력

# 윈도우를 모니터 중앙에 위치시키기
screen_res = 1920, 1080  # 일반적인 모니터 해상도, 필요시 변경
win_w, win_h = fixed_size  # 윈도우 크기
center_x = screen_res[0] // 2 - win_w // 2  # 중앙 x좌표
center_y = screen_res[1] // 2 - win_h // 2  # 중앙 y좌표
cv2.moveWindow('Original', center_x - win_w - 20, center_y)  # 원본 윈도우 위치
cv2.moveWindow('Rotated + Scaled + Translated', center_x + win_w + 20, center_y)  # 변환 윈도우 위치

cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
