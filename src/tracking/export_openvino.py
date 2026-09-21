from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.export(
    format="openvino",
    imgsz=640
)

print("OpenVINO FP32 변환 완료")