from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.export(
    format="openvino",
    imgsz=640,
    int8=True,
    data=r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\src\tracking\sportsmot_calib.yaml",
    fraction=0.2
)

print("OpenVINO INT8 변환 완료")