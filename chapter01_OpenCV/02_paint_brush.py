import cv2 as cv  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트

# 이미지 가져오기
img = cv.imread(r'c:/computervision/chapter01_OpenCV/soccer.jpg')
# 이미지 크기 고정 
max_width = 400  # 최대 가로 크기
max_height = 400  # 최대 세로 크기
h, w = img.shape[:2]  # 원본 이미지의 세로, 가로 크기 추출
scale = min(max_width / w, max_height / h)  # 비율 계산 (가로, 세로 중 작은 값 기준)
new_w = int(w * scale)  # 리사이즈할 가로 크기
new_h = int(h * scale)  # 리사이즈할 세로 크기
img = cv.resize(img, (new_w, new_h))  # 이미지 리사이즈

brush_size = 5  # 초기 붓 크기
max_brush = 15  # 붓 크기 최대값
min_brush = 1   # 붓 크기 최소값
painting = False  # 드래그 상태(붓질 중인지 여부)
color = (255, 0, 0)  # 기본 파란색 (BGR)

# 마우스 이벤트 콜백 함수 정의
def draw(event, x, y, flags, param):
    global painting, color, img, brush_size
    if event == cv.EVENT_LBUTTONDOWN:  # 좌클릭 시작
        painting = True  # 붓질 시작
        color = (255, 0, 0)  # 파란색 선택
        cv.circle(img, (x, y), brush_size, color, -1)  # 현재 위치에 붓 크기만큼 원 그림
    elif event == cv.EVENT_RBUTTONDOWN:  # 우클릭 시작
        painting = True  # 붓질 시작
        color = (0, 0, 255)  # 빨간색 선택
        cv.circle(img, (x, y), brush_size, color, -1)  # 현재 위치에 붓 크기만큼 원 그림
    elif event == cv.EVENT_MOUSEMOVE and painting:  # 마우스 이동 중 붓질
        cv.circle(img, (x, y), brush_size, color, -1)  # 드래그 중이면 계속 그림
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:  # 클릭 해제
        painting = False  # 붓질 종료

cv.namedWindow('Paint')  # 'Paint'라는 이름의 창 생성
cv.setMouseCallback('Paint', draw)  # 마우스 이벤트 콜백 등록

while True:
    cv.imshow('Paint', img)  # 이미지를 창에 표시
    key = cv.waitKey(1) & 0xFF  # 키 입력 대기 및 값 읽기
    if key == ord('q'):  # q 키 입력 시
        break  # 루프 종료 및 창 닫기
    elif key == ord('+') or key == ord('='):  # + 또는 = 키 입력 시
        brush_size = min(max_brush, brush_size + 1)  # 붓 크기 1 증가 (최대 15)
    elif key == ord('-'):  # - 키 입력 시
        brush_size = max(min_brush, brush_size - 1)  # 붓 크기 1 감소 (최소 1)

cv.destroyAllWindows()  # 모든 OpenCV 창 닫기
