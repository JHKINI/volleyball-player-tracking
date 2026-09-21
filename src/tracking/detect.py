from ultralytics import YOLO
import cv2


MODEL_PATH = "yolo26s.pt"
VIDEO_PATH = "../../data/videos/volleyball.mp4"


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[0],
        conf=0.4,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    cv2.imshow(
        "YOLO26 Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()