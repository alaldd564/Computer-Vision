import cv2 as cv  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트

# 이미지 파일을 읽어옴
img = cv.imread(r'c:/computervision/chapter01_OpenCV/soccer.jpg')# 이미지를 BGR 형식으로 읽어옴
# 이미지가 정상적으로 읽혔는지 확인
if img is None:
    print('이미지를 찾을 수 없습니다.')  # 이미지 파일이 없으면 에러 메시지 출력
    exit()  # 프로그램 종료

# 최대 출력 크기 지정 (가로 400, 세로 400) (없이 실행하면 너무 크게 출력해져서 변환된 사진이 짤림)
max_width = 400  # 최대 가로 크기
max_height = 400  # 최대 세로 크기
h, w = img.shape[:2]  # 원본 이미지의 세로, 가로 크기 추출
scale = min(max_width / w, max_height / h)  # 비율 계산 (가로, 세로 중 작은 값 기준)
new_w = int(w * scale)  # 리사이즈할 가로 크기
new_h = int(h * scale)  # 리사이즈할 세로 크기
img_resized = cv.resize(img, (new_w, new_h))  # 이미지 리사이즈

# 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)  # BGR 이미지를 그레이스케일로 변환
# 그레이스케일 이미지를 BGR로 변환 (hstack을 위해 채널 맞춤)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 원본과 그레이스케일 이미지를 가로로 연결
result = np.hstack((img_resized, gray_bgr))  # 두 이미지를 가로로 이어붙임

# 결과 이미지를 화면에 표시
cv.imshow('Original | Grayscale', result)  # 창 이름과 이미지를 지정하여 출력
# 아무 키나 누를 때까지 대기
cv.waitKey(0)  # 키 입력 대기
# 모든 OpenCV 창 닫기
cv.destroyAllWindows()  # 모든 창 닫기
