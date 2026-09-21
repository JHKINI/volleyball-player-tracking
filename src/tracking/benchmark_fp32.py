from pathlib import Path
import cv2
import time
import statistics
import psutil
import threading
from ultralytics import YOLO


# =========================
# 경로
# =========================

IMAGE_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\data\sportmot\val\val\v_9MHDmAMxO5I_c009\img1"
)

MODEL_PATH = "yolo26s.pt"


# =========================
# CPU 측정 함수
# =========================

cpu_samples = []
stop_cpu_monitor = threading.Event()


def monitor_cpu():
    while not stop_cpu_monitor.is_set():
        cpu = psutil.cpu_percent(interval=0.2)
        cpu_samples.append(cpu)


# =========================
# 모델 로드
# =========================

model = YOLO(MODEL_PATH)

image_paths = sorted([
    p for p in IMAGE_DIR.glob("*.jpg")
    if not p.name.startswith("._")
])

print(f"총 프레임 수: {len(image_paths)}")


# =========================
# Warm-up
# =========================

print("Warm-up 시작...")

for image_path in image_paths[:10]:
    frame = cv2.imread(str(image_path))

    if frame is not None:
        model.predict(
            frame,
            conf=0.4,
            classes=[0],
            verbose=False
        )

print("Warm-up 완료")


# =========================
# CPU 모니터링 시작
# =========================

cpu_samples.clear()
stop_cpu_monitor.clear()

cpu_thread = threading.Thread(
    target=monitor_cpu,
    daemon=True
)

cpu_thread.start()


# =========================
# Benchmark
# =========================

times = []

total_start = time.perf_counter()

for image_path in image_paths:

    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    start = time.perf_counter()

    model.predict(
        frame,
        conf=0.4,
        classes=[0],
        verbose=False
    )

    end = time.perf_counter()

    times.append(end - start)


total_time = time.perf_counter() - total_start


# =========================
# CPU 모니터링 종료
# =========================

stop_cpu_monitor.set()
cpu_thread.join(timeout=1)

avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
max_cpu = max(cpu_samples) if cpu_samples else 0


# =========================
# 결과 계산
# =========================

avg_time = statistics.mean(times)
fps = 1 / avg_time


# =========================
# 결과 출력
# =========================

print()
print("=" * 50)
print(" PyTorch FP32 Benchmark")
print("=" * 50)

print(f"프레임 수          : {len(times)}")
print(f"총 처리 시간       : {total_time:.2f} sec")
print(f"평균 추론 시간     : {avg_time * 1000:.2f} ms")
print(f"FPS                : {fps:.2f}")
print(f"평균 CPU 점유율    : {avg_cpu:.2f}%")
print(f"최대 CPU 점유율    : {max_cpu:.2f}%")

print("=" * 50)