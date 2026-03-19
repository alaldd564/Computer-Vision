import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # numpy 임포트
import os  # OS 경로 처리를 위한 os 임포트

# 카메라 파라미터 설정
f = 700.0  # focal length (초점 거리)
B = 0.12   # baseline (두 카메라 사이 거리, m)

# ROI 영역 정의
rois = {
    "Painting": (55, 50, 130, 110),  # 그림 영역
    "Frog": (90, 265, 230, 95),      # 개구리 영역
    "Teddy": (310, 35, 115, 90)      # 곰인형 영역
}

# 이미지 경로 설정
left_path = os.path.join(os.path.dirname(__file__), 'images/left.png')  # 좌측 이미지 경로
right_path = os.path.join(os.path.dirname(__file__), 'images/right.png')  # 우측 이미지 경로

# 이미지 읽기
left_img = cv2.imread(left_path)  # 좌측 이미지 읽기
right_img = cv2.imread(right_path)  # 우측 이미지 읽기

# 이미지가 없을 경우 예외 처리
if left_img is None or right_img is None:
    raise FileNotFoundError('좌/우 이미지를 찾을 수 없습니다.')  # 이미지 없을 때 예외 처리

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)  # 좌측 그레이스케일 변환
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)  # 우측 그레이스케일 변환

# StereoBM으로 disparity map 계산
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # StereoBM 객체 생성
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # disparity 계산 및 스케일 조정

# Disparity > 0인 픽셀만 depth 계산
valid_mask = disparity > 0  # 유효 disparity 마스크

# depth map 초기화 및 계산
depth_map = np.zeros_like(disparity)  # depth map 초기화
depth_map[valid_mask] = f * B / disparity[valid_mask]  # depth 계산

# ROI별 평균 disparity, depth 계산
results = {}  # 결과 저장 딕셔너리
for name, (x, y, w, h) in rois.items():  # 각 ROI에 대해 반복
    roi_disp = disparity[y:y+h, x:x+w]  # ROI 내 disparity 추출
    roi_depth = depth_map[y:y+h, x:x+w]  # ROI 내 depth 추출
    roi_mask = roi_disp > 0  # 유효 disparity 마스크
    mean_disp = np.mean(roi_disp[roi_mask]) if np.any(roi_mask) else np.nan  # 평균 disparity 계산
    mean_depth = np.mean(roi_depth[roi_mask]) if np.any(roi_mask) else np.nan  # 평균 depth 계산
    results[name] = {'mean_disp': mean_disp, 'mean_depth': mean_depth}  # 결과 저장

# 가장 가까운/먼 ROI 찾기
closest_roi = min(results, key=lambda k: results[k]['mean_depth'])  # 가장 가까운 ROI (depth가 가장 작음)
farthest_roi = max(results, key=lambda k: results[k]['mean_depth'])  # 가장 먼 ROI (depth가 가장 큼)

# 결과 출력
print('ROI별 평균 disparity, depth:')  # ROI별 평균 disparity, depth 출력
for name, vals in results.items():
    print(f"{name}: disparity={vals['mean_disp']:.2f}, depth={vals['mean_depth']:.2f}")  # 각 ROI별 값 출력
print(f"\n가장 가까운 ROI: {closest_roi}")  # 가장 가까운 ROI 출력
print(f"가장 먼 ROI: {farthest_roi}")  # 가장 먼 ROI 출력

# Disparity map 시각화
disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # 시각화용 배열 초기화
if np.any(valid_mask):  # 유효 disparity가 있을 때
    d_min = np.nanpercentile(disparity[valid_mask], 5)  # 최소값 계산
    d_max = np.nanpercentile(disparity[valid_mask], 95)  # 최대값 계산
    disp_scaled = (disparity - d_min) / (d_max - d_min)  # 정규화
    disp_scaled = np.clip(disp_scaled, 0, 1)  # 범위 제한
    disp_vis[valid_mask] = (disp_scaled[valid_mask] * 255).astype(np.uint8)  # 0~255 변환
color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # 컬러맵 적용

# 결과 시각화
cv2.imshow('Original', left_img)  # 원본 이미지 출력
cv2.imshow('Disparity map', color_disp)  # disparity map 출력
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
import cv2
import numpy as np
import os

# 파라미터
f = 700.0  # focal length
B = 0.12   # baseline (m)

# ROI 영역 정의
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 이미지 경로
left_path = os.path.join(os.path.dirname(__file__), 'images/left.png')
right_path = os.path.join(os.path.dirname(__file__), 'images/right.png')

left_img = cv2.imread(left_path)
right_img = cv2.imread(right_path)

if left_img is None or right_img is None:
    raise FileNotFoundError('좌/우 이미지를 찾을 수 없습니다.')

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# StereoBM으로 disparity map 계산
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Disparity > 0인 픽셀만 depth 계산
valid_mask = disparity > 0

depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = f * B / disparity[valid_mask]

# ROI별 평균 disparity, depth 계산
results = {}
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_mask = roi_disp > 0
    mean_disp = np.mean(roi_disp[roi_mask]) if np.any(roi_mask) else np.nan
    mean_depth = np.mean(roi_depth[roi_mask]) if np.any(roi_mask) else np.nan
    results[name] = {'mean_disp': mean_disp, 'mean_depth': mean_depth}

# 가장 가까운/먼 ROI 찾기
closest_roi = min(results, key=lambda k: results[k]['mean_depth'])
farthest_roi = max(results, key=lambda k: results[k]['mean_depth'])

print('ROI별 평균 disparity, depth:')
for name, vals in results.items():
    print(f"{name}: disparity={vals['mean_disp']:.2f}, depth={vals['mean_depth']:.2f}")
print(f"\n가장 가까운 ROI: {closest_roi}")
print(f"가장 먼 ROI: {farthest_roi}")

# Disparity map 시각화
# 정규화
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
if np.any(valid_mask):
    d_min = np.nanpercentile(disparity[valid_mask], 5)
    d_max = np.nanpercentile(disparity[valid_mask], 95)
    disp_scaled = (disparity - d_min) / (d_max - d_min)
    disp_scaled = np.clip(disp_scaled, 0, 1)
    disp_vis[valid_mask] = (disp_scaled[valid_mask] * 255).astype(np.uint8)

color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# 결과 시각화
cv2.imshow('Original', left_img)
cv2.imshow('Disparity map', color_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
