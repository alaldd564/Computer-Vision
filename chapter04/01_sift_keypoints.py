import cv2  # OpenCV 라이브러리 임포트 (이미지 처리 기능 제공)
import matplotlib.pyplot as plt  # Matplotlib 임포트 (시각화 기능 제공)
import numpy as np  # NumPy 임포트 (수치 연산 기능 제공)

# 첫 번째 경로에서 이미지 로드 시도
img_path = cv2.imread('chapter04/mot_color70.jpg')
# 이미지 로드 실패 시 현재 디렉토리에서 이미지 로드 시도
if img_path is None:
    img_path = cv2.imread('mot_color70.jpg')

# OpenCV의 BGR 형식 이미지를 matplotlib용 RGB 형식으로 변환
img_rgb = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)

# 이미지를 그레이스케일로 변환 (SIFT 알고리즘은 그레이스케일 이미지에서 작동)
img_gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

# ===== SIFT 특징점 검출 =====
# 다양한 nfeatures 값으로 비교를 위해 여러 SIFT 객체 생성

# 1. 기본 SIFT 객체 생성 (특징점 개수 제한 없음)
sift = cv2.SIFT_create()
# 기본 SIFT로 그레이스케일 이미지에서 특징점과 묘사자(descriptor) 검출
keypoints_default, descriptors_default = sift.detectAndCompute(img_gray, None)

# 2. SIFT 객체 생성 (nfeatures=500으로 최대 특징점 개수 제한)
sift_500 = cv2.SIFT_create(nfeatures=500)
# 500개 제한의 SIFT로 특징점과 묘사자 검출
keypoints_500, descriptors_500 = sift_500.detectAndCompute(img_gray, None)

# 3. SIFT 객체 생성 (nfeatures=200으로 최대 특징점 개수 제한)
sift_200 = cv2.SIFT_create(nfeatures=200)
# 200개 제한의 SIFT로 특징점과 묘사자 검출
keypoints_200, descriptors_200 = sift_200.detectAndCompute(img_gray, None)

# 기본 SIFT로 검출된 특징점 개수 출력
print(f"기본 SIFT - 특징점 개수: {len(keypoints_default)}")
# 500개 제한 SIFT로 검출된 특징점 개수 출력
print(f"nfeatures=500 - 특징점 개수: {len(keypoints_500)}")
# 200개 제한 SIFT로 검출된 특징점 개수 출력
print(f"nfeatures=200 - 특징점 개수: {len(keypoints_200)}")

# ===== 특징점 시각화 =====
# 특징점의 방향과 크기를 표시하기 위해 DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 사용

# 1. 기본 SIFT 결과를 이미지에 그리기
img_keypoints_default = cv2.drawKeypoints(
    img_gray,  # 그리기를 할 입력 그레이스케일 이미지
    keypoints_default,  # 그릴 특징점 리스트
    None,  # 출력 이미지 (None으로 설정하면 새로 생성)
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 특징점의 방향과 크기를 표시하는 플래그
)

# 2. nfeatures=500 결과를 이미지에 그리기
img_keypoints_500 = cv2.drawKeypoints(
    img_gray,  # 입력 그레이스케일 이미지
    keypoints_500,  # 특징점 리스트 (500개 제한)
    None,  # 출력 이미지
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 특징점 시각화 옵션
)

# 3. nfeatures=200 결과를 이미지에 그리기
img_keypoints_200 = cv2.drawKeypoints(
    img_gray,  # 입력 그레이스케일 이미지
    keypoints_200,  # 특징점 리스트 (200개 제한)
    None,  # 출력 이미지
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 특징점 시각화 옵션
)

# ===== matplotlib을 이용한 시각화 =====

# (1) 원본 이미지 vs 기본 SIFT 특징점 비교 시각화
# 1행 2열의 서브플롯 생성 (크기: 14x6)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 첫 번째 서브플롯에 원본 RGB 이미지 표시
axes[0].imshow(img_rgb)
# 첫 번째 서브플롯의 제목 설정
axes[0].set_title('Original Image (mot_color70.jpg)', fontsize=12)
# 첫 번째 서브플롯의 축 제거
axes[0].axis('off')

# 기본 SIFT 특징점 이미지를 BGR에서 RGB로 변환
img_keypoints_default_rgb = cv2.cvtColor(img_keypoints_default, cv2.COLOR_BGR2RGB)
# 두 번째 서브플롯에 변환된 특징점 이미지 표시
axes[1].imshow(img_keypoints_default_rgb)
# 두 번째 서브플롯의 제목 설정 (특징점 개수 표시)
axes[1].set_title(f'SIFT Keypoints (Default: {len(keypoints_default)} points)', fontsize=12)
# 두 번째 서브플롯의 축 제거
axes[1].axis('off')

# 서브플롯 간의 간격 자동 조정
plt.tight_layout()
# 그래프를 PNG 파일로 저장 (해상도: 150 dpi)
plt.savefig('chapter04/01_sift_default.png', dpi=150, bbox_inches='tight')
# 그래프 화면 표시
plt.show()


# (2) nfeatures 비교 - 3개 결과를 나란히 출력
# 1행 3열의 서브플롯 생성 (크기: 18x6)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 기본 SIFT 결과 이미지를 BGR에서 RGB로 변환
img_kp_default_rgb = cv2.cvtColor(img_keypoints_default, cv2.COLOR_BGR2RGB)
# nfeatures=500 결과 이미지를 BGR에서 RGB로 변환
img_kp_500_rgb = cv2.cvtColor(img_keypoints_500, cv2.COLOR_BGR2RGB)
# nfeatures=200 결과 이미지를 BGR에서 RGB로 변환
img_kp_200_rgb = cv2.cvtColor(img_keypoints_200, cv2.COLOR_BGR2RGB)

# 첫 번째 서브플롯에 기본 SIFT 결과 표시
axes[0].imshow(img_kp_default_rgb)
# 첫 번째 서브플롯의 제목 설정 (특징점 개수 표시)
axes[0].set_title(f'Default SIFT\n({len(keypoints_default)} keypoints)', fontsize=12)
# 첫 번째 서브플롯의 축 제거
axes[0].axis('off')

# 두 번째 서브플롯에 nfeatures=500 결과 표시
axes[1].imshow(img_kp_500_rgb)
# 두 번째 서브플롯의 제목 설정 (특징점 개수 표시)
axes[1].set_title(f'SIFT (nfeatures=500)\n({len(keypoints_500)} keypoints)', fontsize=12)
# 두 번째 서브플롯의 축 제거
axes[1].axis('off')

# 세 번째 서브플롯에 nfeatures=200 결과 표시
axes[2].imshow(img_kp_200_rgb)
# 세 번째 서브플롯의 제목 설정 (특징점 개수 표시)
axes[2].set_title(f'SIFT (nfeatures=200)\n({len(keypoints_200)} keypoints)', fontsize=12)
# 세 번째 서브플롯의 축 제거
axes[2].axis('off')

# 서브플롯 간의 간격 자동 조정
plt.tight_layout()
# 비교 그래프를 PNG 파일로 저장 (해상도: 150 dpi)
plt.savefig('chapter04/01_sift_comparison.png', dpi=150, bbox_inches='tight')
# 그래프 화면 표시
plt.show()

# ===== 특징점 정보 출력 =====
# 구분선과 함께 특징점 정보 출력 섹션 시작
print("\n===== 특징점 정보 =====")
# 첫 5개 특징점 정보를 출력할 섹션 시작
print(f"\n첫 5개 특징점 (기본 SIFT):")
# 기본 SIFT의 첫 5개 특징점에 대해 반복
for i, kp in enumerate(keypoints_default[:5]):
    # 각 특징점의 위치(x, y), 크기, 각도(방향) 정보 출력
    print(f"  {i+1}. 위치: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f}), "
          f"크기: {kp.size:.2f}, 각도: {kp.angle:.2f}°")

# 3가지 방식의 특징점 검출 결과를 비교하는 섹션 시작
print("\n특징점 검출 결과 비교:")
# 기본 SIFT의 특징점 개수 출력
print(f"  기본 SIFT: {len(keypoints_default)} 개")
# nfeatures=500의 특징점 개수와 기본 방식 대비 비율 출력
print(f"  nfeatures=500: {len(keypoints_500)} 개 (원래 대비 {len(keypoints_500)/len(keypoints_default)*100:.1f}%)")
# nfeatures=200의 특징점 개수와 기본 방식 대비 비율 출력
print(f"  nfeatures=200: {len(keypoints_200)} 개 (원래 대비 {len(keypoints_200)/len(keypoints_default)*100:.1f}%)")
