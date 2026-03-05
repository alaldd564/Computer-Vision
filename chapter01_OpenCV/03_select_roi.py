import cv2 as cv  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트

img = cv.imread(r'c:/computervision/chapter01_OpenCV/soccer.jpg') # 이미지를 불러옴
if img is None:
    print('이미지를 찾을 수 없습니다.')  # 이미지가 없으면 에러 메시지 출력
    exit()  # 프로그램 종료

# 이미지 크기 고정 
max_width = 400  # 최대 가로 크기
max_height = 400  # 최대 세로 크기
h, w = img.shape[:2]  # 원본 이미지의 세로, 가로 크기 추출
scale = min(max_width / w, max_height / h)  # 비율 계산 (가로, 세로 중 작은 값 기준)
new_w = int(w * scale)  # 리사이즈할 가로 크기
new_h = int(h * scale)  # 리사이즈할 세로 크기
img = cv.resize(img, (new_w, new_h))  # 이미지 리사이즈

clone = img.copy()  # 원본 이미지 복사본 생성
roi = None  # ROI(관심영역) 변수
selecting = False  # 영역 선택 중 여부
start_point = (-1, -1)  # 선택 시작점
end_point = (-1, -1)    # 선택 끝점

# 마우스 이벤트 콜백 함수 정의
def select_roi(event, x, y, flags, param):
    global start_point, end_point, selecting, roi, img
    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 좌클릭 시작
        start_point = (x, y)  # 시작점 저장
        selecting = True  # 선택 중 상태로 변경
        end_point = (x, y)  # 끝점도 초기화
    elif event == cv.EVENT_MOUSEMOVE and selecting:  # 드래그 중
        end_point = (x, y)  # 끝점 갱신
    elif event == cv.EVENT_LBUTTONUP:  # 마우스 좌클릭 해제
        end_point = (x, y)  # 끝점 저장
        selecting = False  # 선택 종료
        # ROI 추출 (numpy 슬라이싱)
        x1, y1 = start_point  # 시작점 좌표
        x2, y2 = end_point    # 끝점 좌표
        x1, x2 = sorted([x1, x2])  # 좌표 정렬
        y1, y2 = sorted([y1, y2])  # 좌표 정렬
        roi = clone[y1:y2, x1:x2]  # ROI 추출

cv.namedWindow('Image')  # 이미지 창 생성
cv.setMouseCallback('Image', select_roi)  # 마우스 이벤트 콜백 등록

while True:
    display_img = clone.copy()  # 이미지 복사본 준비
    if selecting or (start_point != (-1, -1) and end_point != (-1, -1)):
        # 드래그 중 또는 선택 완료 시 사각형 표시
        cv.rectangle(display_img, start_point, end_point, (0, 255, 0), 2)  # 선택 영역 시각화
    cv.imshow('Image', display_img)  # 이미지 출력
    if roi is not None:
        cv.imshow('ROI', roi)  # ROI가 있으면 별도 창에 출력
    key = cv.waitKey(1) & 0xFF  # 키 입력 대기 및 값 읽기
    if key == ord('q'):  # q 키로 종료
        break  # 루프 종료 및 창 닫기
    elif key == ord('r'):  # r 키로 선택 리셋
        start_point = (-1, -1)  # 시작점 초기화
        end_point = (-1, -1)    # 끝점 초기화
        roi = None  # ROI 초기화
        clone = img.copy()  # 이미지 복원
        cv.destroyWindow('ROI')  # ROI 창 닫기
    elif key == ord('s') and roi is not None:  # s 키로 ROI 저장
        cv.imwrite('roi.jpg', roi)  # ROI 이미지를 파일로 저장
        print('ROI가 roi.jpg로 저장되었습니다.')  # 저장 메시지 출력

cv.destroyAllWindows()  # 모든 창 닫기
