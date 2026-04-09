# 운영체제 경로 처리를 위한 os 모듈을 가져옵니다.
import os
# 영상 캡처 및 화면 표시를 위한 OpenCV를 가져옵니다.
import cv2
# MediaPipe 이미지 객체 생성을 위해 mediapipe를 가져옵니다.
import mediapipe as mp
# MediaPipe Tasks의 파이썬 베이스 API를 가져옵니다.
from mediapipe.tasks import python
# FaceLandmarker 비전 태스크 API를 가져옵니다.
from mediapipe.tasks.python import vision

# 현재 파일과 같은 폴더에 있는 모델 파일 경로를 만듭니다.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


# 프로그램의 메인 실행 함수를 정의합니다.
def main() -> None:
    # 모델 파일이 존재하지 않으면 안내 메시지를 출력합니다.
    if not os.path.exists(MODEL_PATH):
        # 누락된 모델 파일 경로를 사용자에게 보여줍니다.
        print("모델 파일이 없습니다:", MODEL_PATH)
        # 모델 파일을 어디에 둬야 하는지 안내합니다.
        print("face_landmarker.task 파일을 chapter06 폴더에 두고 다시 실행하세요.")
        # 모델이 없으므로 함수를 종료합니다.
        return

    # 얼굴 랜드마커 옵션 객체를 생성합니다.
    options = vision.FaceLandmarkerOptions(
        # 로컬 모델 파일 경로를 베이스 옵션에 설정합니다.
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        # 한 프레임에서 최대 1개의 얼굴만 처리합니다.
        num_faces=1,
        # 블렌드셰이프 출력은 사용하지 않습니다.
        output_face_blendshapes=False,
        # 얼굴 변환 행렬 출력은 사용하지 않습니다.
        output_facial_transformation_matrixes=False,
    )

    # 기본 카메라(인덱스 0)를 엽니다.
    cap = cv2.VideoCapture(0)
    # 카메라를 열지 못하면 오류 메시지를 출력합니다.
    if not cap.isOpened():
        # 웹캠 접근 실패 메시지를 출력합니다.
        print("웹캠을 열 수 없습니다.")
        # 카메라가 없으므로 함수를 종료합니다.
        return

    # 옵션으로 FaceLandmarker를 생성하고 블록 종료 시 자동 해제합니다.
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # 종료 조건(ESC) 전까지 프레임을 계속 처리합니다.
        while True:
            # 카메라에서 한 프레임을 읽습니다.
            ret, frame = cap.read()
            # 프레임 읽기에 실패하면 루프를 종료합니다.
            if not ret:
                # 프레임 획득 실패 메시지를 출력합니다.
                print("카메라에서 영상을 읽을 수 없습니다.")
                # 더 이상 처리할 수 없어 반복을 중단합니다.
                break

            # OpenCV의 BGR 프레임을 MediaPipe 입력용 RGB로 변환합니다.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # RGB numpy 배열을 MediaPipe Image 객체로 감쌉니다.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # 현재 프레임에서 얼굴 랜드마크를 검출합니다.
            result = landmarker.detect(mp_image)

            # 랜드마크 결과가 있으면 점을 그립니다.
            if result.face_landmarks:
                # 좌표 변환을 위해 프레임의 높이와 너비를 가져옵니다.
                h, w, _ = frame.shape
                # 검출된 각 얼굴의 랜드마크 집합을 순회합니다.
                for landmarks in result.face_landmarks:
                    # 해당 얼굴의 모든 랜드마크 포인트를 순회합니다.
                    for lm in landmarks:
                        # 정규화된 x 좌표를 픽셀 좌표로 변환합니다.
                        x = int(lm.x * w)
                        # 정규화된 y 좌표를 픽셀 좌표로 변환합니다.
                        y = int(lm.y * h)
                        # 유효한 화면 범위 안의 점만 그립니다.
                        if 0 <= x < w and 0 <= y < h:
                            # 랜드마크 위치에 초록색 점을 표시합니다.
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 랜드마크가 그려진 프레임을 화면에 표시합니다.
            cv2.imshow("MediaPipe Face Landmarks", frame)
            # ESC(27) 키가 눌리면 루프를 종료합니다.
            if cv2.waitKey(1) & 0xFF == 27:
                # 사용자 종료 요청에 따라 반복을 중단합니다.
                break

    # 카메라 자원을 해제합니다.
    cap.release()
    # OpenCV 창을 모두 닫습니다.
    cv2.destroyAllWindows()


# 이 파일이 직접 실행될 때 main 함수를 호출합니다.
if __name__ == "__main__":
    # 메인 로직을 실행합니다.
    main()
