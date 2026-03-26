import cv2  # OpenCV 라이브러리 임포트 (컴퓨터 비전 기능 제공)
import matplotlib.pyplot as plt  # Matplotlib 임포트 (시각화 기능 제공)
import numpy as np  # NumPy 임포트 (수치 연산 기능 제공)

# 첫 번째 이미지 파일 경로 설정
img1_path = 'chapter04/mot_color70.jpg'
# 두 번째 이미지 파일 경로 설정
img2_path = 'chapter04/mot_color83.jpg'

# 첫 번째 이미지 읽기 (BGR 형식으로 로드)
img1 = cv2.imread(img1_path)
# 두 번째 이미지 읽기 (BGR 형식으로 로드)
img2 = cv2.imread(img2_path)

# 첫 번째 이미지 로드 실패 시 현재 디렉토리에서 재시도
if img1 is None:
    img1 = cv2.imread('mot_color70.jpg')
# 두 번째 이미지 로드 실패 시 현재 디렉토리에서 재시도
if img2 is None:
    img2 = cv2.imread('mot_color83.jpg')

# 두 이미지 모두 로드 실패 시 에러 메시지 출력 및 프로그램 종료
if img1 is None or img2 is None:
    raise FileNotFoundError('이미지 파일을 찾을 수 없습니다.')

# 첫 번째 이미지를 BGR에서 그레이스케일로 변환 (SIFT용)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 두 번째 이미지를 BGR에서 그레이스케일로 변환 (SIFT용)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 특징 추출자 객체 생성
sift = cv2.SIFT_create()

# 첫 번째 이미지에서 특징점과 디스크립터 검출 및 계산
gkp1, des1 = sift.detectAndCompute(img1_gray, None)
# 두 번째 이미지에서 특징점과 디스크립터 검출 및 계산
gkp2, des2 = sift.detectAndCompute(img2_gray, None)

# 두 이미지에서 검출된 특징점 개수 출력
print(f"이미지1 특징점: {len(gkp1)}개, 이미지2 특징점: {len(gkp2)}개")

# ========== 매칭 방법 1: BFMatcher (crossCheck) ==========
# BFMatcher 객체 생성 (L2 거리 사용, crossCheck=True로 상호 매칭만 허용)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# BF 알고리즘으로 두 이미지의 디스크립터 매칭 수행
matches_bf = bf.match(des1, des2)
# 매칭 결과를 거리 기준으로 오름차순 정렬 (더 좋은 매칭이 먼저 나옴)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# ========== 매칭 방법 2: FLANN + knnMatch + ratio test ==========
# FLANN KDTree 인덱스 상수 정의
FLANN_INDEX_KDTREE = 1
# KDTree 알고리즘 파라미터 설정 (trees=5: 5개의 트리 사용)
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# FLANN 검색 파라미터 설정 (checks=50: 최대 50번의 비교 수행)
search_params = dict(checks=50)
# FLANN 기반 매처 객체 생성
flann = cv2.FlannBasedMatcher(index_params, search_params)

# knnMatch로 각 쿼리 디스크립터마다 k=2개의 최근린 매치 찾기
matches_knn = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test를 적용한 좋은 매칭 필터링
good_matches = []
# knnMatch 결과의 각 매칭 쌍(m, n)에 대해 순회
for m, n in matches_knn:
    # 첫 번째 매칭의 거리가 두 번째 매칭의 거리의 0.75배보다 작으면 좋은 매칭 선택
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# BF 매처로 찾은 매칭 개수 출력
print(f"BFMatcher 매칭 개수: {len(matches_bf)}")
# FLANN + ratio test로 찾은 매칭 개수 출력
print(f"FLANN + ratio test 매칭 개수: {len(good_matches)}")

# ========== 시각화 ==========
# 1. BFMatcher 결과 시각화
# cv2.drawMatches로 두 이미지와 매칭된 특징점을 시각화 (상위 30개)
img_bf = cv2.drawMatches(img1, gkp1, img2, gkp2, matches_bf[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 2. FLANN + ratio test 결과 시각화
# cv2.drawMatches로 두 이미지와 매칭된 특징점을 시각화 (상위 30개)
img_flann = cv2.drawMatches(img1, gkp1, img2, gkp2, good_matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# BFMatcher 결과 이미지를 BGR에서 RGB로 변환 (matplotlib용)
img_bf_rgb = cv2.cvtColor(img_bf, cv2.COLOR_BGR2RGB)
# FLANN 결과 이미지를 BGR에서 RGB로 변환 (matplotlib용)
img_flann_rgb = cv2.cvtColor(img_flann, cv2.COLOR_BGR2RGB)

# 1행 2열의 서브플롯 생성 (크기: 18x8)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# 첫 번째 서브플롯에 BFMatcher 결과 이미지 표시
axes[0].imshow(img_bf_rgb)
# 첫 번째 서브플롯의 제목 설정
axes[0].set_title('BFMatcher (Top 30 matches)', fontsize=14)
# 첫 번째 서브플롯의 축 제거
axes[0].axis('off')
# 두 번째 서브플롯에 FLANN 결과 이미지 표시
axes[1].imshow(img_flann_rgb)
# 두 번째 서브플롯의 제목 설정
axes[1].set_title('FLANN + Ratio Test (Top 30 matches)', fontsize=14)
# 두 번째 서브플롯의 축 제거
axes[1].axis('off')
# 서브플롯 간의 간격 자동 조정
plt.tight_layout()
# 그래프를 PNG 파일로 저장 (해상도: 150 dpi)
plt.savefig('chapter04/02_sift_feature_matching.png', dpi=150, bbox_inches='tight')
# 그래프 화면 표시
plt.show()

# 매칭 정보 출력 섹션 제목
print("\n===== 매칭 정보 (BFMatcher 상위 5개) =====")
# BFMatcher의 상위 5개 매칭에 대해 순회
for i, m in enumerate(matches_bf[:5]):
    # 각 매칭의 인덱스, 두 이미지의 특징점 위치, 거리 출력
    print(f"{i+1}. 이미지1: {gkp1[m.queryIdx].pt}, 이미지2: {gkp2[m.trainIdx].pt}, 거리: {m.distance:.2f}")

# FLANN + ratio test 매칭 정보 출력 섹션 제목
print("\n===== 매칭 정보 (FLANN + Ratio Test 상위 5개) =====")
# FLANN + ratio test의 상위 5개 매칭에 대해 순회
for i, m in enumerate(good_matches[:5]):
    # 각 매칭의 인덱스, 두 이미지의 특징점 위치, 거리 출력
    print(f"{i+1}. 이미지1: {gkp1[m.queryIdx].pt}, 이미지2: {gkp2[m.trainIdx].pt}, 거리: {m.distance:.2f}")
