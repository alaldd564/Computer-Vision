
# OpenCV 라이브러리 import
import cv2
# numpy 라이브러리 import (행렬, 배열 연산)
import numpy as np
# matplotlib 라이브러리 import (시각화)
import matplotlib.pyplot as plt

# 이미지 파일 경로 지정 (샘플: img1.jpg, img2.jpg)
img1_path = 'chapter04/img1.jpg'  # 첫 번째 이미지 경로
img2_path = 'chapter04/img2.jpg'  # 두 번째 이미지 경로

# 이미지 파일 읽기 (컬러)
img1 = cv2.imread(img1_path)  # 첫 번째 이미지 읽기
img2 = cv2.imread(img2_path)  # 두 번째 이미지 읽기

# 만약 chapter04 폴더에 없으면 현재 폴더에서 다시 시도
if img1 is None:
    img1 = cv2.imread('img1.jpg')  # 대체 경로 시도
if img2 is None:
    img2 = cv2.imread('img2.jpg')  # 대체 경로 시도

# 이미지가 모두 정상적으로 읽혔는지 확인
if img1 is None or img2 is None:
    raise FileNotFoundError('img1.jpg, img2.jpg 파일을 찾을 수 없습니다.')  # 파일 없으면 에러 발생

# 이미지를 그레이스케일로 변환 (SIFT는 그레이스케일 사용)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 첫 번째 이미지 변환
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 두 번째 이미지 변환

# SIFT 객체 생성 (특징점 검출기)
sift = cv2.SIFT_create()  # SIFT 알고리즘 객체 생성
# 첫 번째 이미지에서 특징점과 디스크립터 추출
kp1, des1 = sift.detectAndCompute(img1_gray, None)
# 두 번째 이미지에서 특징점과 디스크립터 추출
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFMatcher 객체 생성 (브루트포스 매칭)
bf = cv2.BFMatcher()  # 디스크립터 매칭 객체 생성
# knnMatch로 두 이미지의 디스크립터 매칭 (k=2: 최근접 2개)
matches = bf.knnMatch(des1, des2, k=2)

# 거리비율 테스트로 좋은 매칭점만 선별 (Lowe's ratio test)
ratio_thresh = 0.7  # 임계값 설정
good_matches = []  # 좋은 매칭점 저장 리스트
for m, n in matches:  # 각 매칭쌍에 대해
    if m.distance < ratio_thresh * n.distance:  # 첫 번째가 두 번째보다 충분히 가까우면
        good_matches.append(m)  # 좋은 매칭점으로 인정

# 전체 매칭 개수와 좋은 매칭 개수 출력
print(f"전체 매칭: {len(matches)}, 좋은 매칭: {len(good_matches)}")

# 좋은 매칭점이 4개 미만이면 호모그래피 계산 불가
if len(good_matches) < 4:
    raise ValueError('호모그래피 계산을 위한 좋은 매칭점이 부족합니다.')

# 좋은 매칭점에서 좌표 추출 (queryIdx: img1, trainIdx: img2)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # img1 좌표
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # img2 좌표

# 호모그래피 행렬 계산 (RANSAC 사용, 이상점 제거)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 변환 행렬 계산
print('호모그래피 행렬:\n', H)  # 행렬 출력

# img1을 호모그래피로 변환하여 파노라마 크기로 워핑
h1, w1 = img1.shape[:2]  # img1 크기
h2, w2 = img2.shape[:2]  # img2 크기
panorama_width = w1 + w2  # 파노라마 너비
panorama_height = max(h1, h2)  # 파노라마 높이

warped_img1 = cv2.warpPerspective(img1, H, (panorama_width, panorama_height))  # img1 변환

# 파노라마 이미지에 img2를 왼쪽에 붙이기 (겹치는 부분 덮어쓰기)
overlay = warped_img1.copy()  # 변환 이미지 복사
overlay[0:h2, 0:w2] = img2  # 왼쪽에 img2 삽입

# 특징점 매칭 결과 시각화 (inlier만 표시)
matches_mask = mask.ravel().tolist()  # inlier/outlier 마스크
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,  # 두 이미지와 매칭점
    matchesMask=matches_mask,  # inlier만 표시
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 특징점만 선으로 표시
)
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)  # BGR->RGB 변환

# 변환된 이미지(파노라마) 시각화용 RGB 변환
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# matplotlib으로 두 결과 이미지를 나란히 출력
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1행 2열
axes[0].imshow(img_matches_rgb)  # 왼쪽: 매칭 결과
axes[0].set_title('SIFT Feature Matching (Inliers)', fontsize=14)  # 제목
axes[0].axis('off')  # 축 숨김
axes[1].imshow(overlay_rgb)  # 오른쪽: 변환 이미지
axes[1].set_title('Warped Image (Alignment Result)', fontsize=14)  # 제목
axes[1].axis('off')  # 축 숨김
plt.tight_layout()  # 여백 자동 조정
plt.savefig('chapter04/03_sift_homography_alignment.png', dpi=150, bbox_inches='tight')  # 파일 저장
plt.show()  # 화면에 출력

# 좋은 매칭점 일부(상위 5개) 정보 출력
print("\n===== 좋은 매칭점 (상위 5개) =====")
for i, m in enumerate(good_matches[:5]):  # 상위 5개만
    print(f"{i+1}. img1: {kp1[m.queryIdx].pt}, img2: {kp2[m.trainIdx].pt}, 거리: {m.distance:.2f}")  # 좌표 및 거리 출력
