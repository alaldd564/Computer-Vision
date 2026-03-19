import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # numpy 임포트
import glob  # 파일 패턴 검색을 위한 glob 임포트
import os  # OS 경로 처리를 위한 os 임포트

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)  # 체크보드의 내부 코너 개수
# 체크보드 한 칸의 실제 크기 (mm)
SQUARE_SIZE = 25.0  # mm, 한 칸의 실제 크기

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 코너 정밀화 조건
# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)  # 3D 좌표 배열 생성
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # 격자 구조 생성
objp *= SQUARE_SIZE  # 실제 크기 반영

# 3D 실제 좌표 저장 리스트
objpoints = []  # 3D 실제 좌표
# 2D 이미지 좌표 저장 리스트
imgpoints = []  # 2D 이미지 좌표

# 이미지 파일 리스트 가져오기
images = glob.glob("images/calibration_images/left*.jpg")  # 체크보드 이미지 파일 리스트

# 이미지 크기 변수
img_size = None  # 이미지 크기 저장 변수

# 1. 체크보드 코너 검출
for fname in images:  # 각 이미지 파일에 대해 반복
    img = cv2.imread(fname)  # 이미지 읽기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    if img_size is None:  # 첫 이미지에서 크기 저장
        img_size = gray.shape[::-1]  # (가로, 세로)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 체크보드 코너 검출
    if ret:  # 검출 성공 시
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 코너 정밀화
        objpoints.append(objp)  # 실제 좌표 저장
        imgpoints.append(corners2)  # 이미지 좌표 저장
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)  # 코너 시각화
        cv2.imshow('Corners', img)  # 이미지 출력
        cv2.waitKey(300)  # 잠시 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기

# 코너 검출 실패 시
if len(objpoints) == 0:  # 코너 검출 실패 시
    print('체크보드 코너를 검출한 이미지가 없습니다.')  # 경고 출력
    exit()  # 프로그램 종료

# 2. 카메라 캘리브레이션
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)  # 카메라 파라미터 계산
print('Camera Matrix K:')  # 내부 행렬 출력
print(K)  # 내부 행렬 값 출력
print('\nDistortion Coefficients:')  # 왜곡 계수 출력
print(dist)  # 왜곡 계수 값 출력

# 3. 왜곡 보정 시각화
for fname in images:  # 각 이미지에 대해 반복
    img = cv2.imread(fname)  # 이미지 읽기
    undistorted = cv2.undistort(img, K, dist, None, K)  # 왜곡 보정
    cv2.imshow('Original', img)  # 원본 이미지 출력
    cv2.imshow('Undistorted', undistorted)  # 보정 이미지 출력
    cv2.waitKey(500)  # 잠시 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
print('\nDistortion Coefficients:')  # 왜곡 계수 출력
