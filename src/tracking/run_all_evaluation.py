from pathlib import Path
import csv
import time
import threading
import psutil
import numpy as np
import cv2
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking"
)

IMAGE_DIR = PROJECT_DIR / "data" / "sportmot" / "val" / "val" / \
    "v_9MHDmAMxO5I_c009" / "img1"

GT_FILE = PROJECT_DIR / "data" / "sportmot" / "val" / "val" / \
    "v_9MHDmAMxO5I_c009" / "gt" / "gt.txt"

RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 모델
PT_MODEL = "yolo26s.pt"

OPENVINO_FP32 = PROJECT_DIR / "yolo26s_openvino_model"
OPENVINO_INT8 = PROJECT_DIR / "yolo26s_int8_openvino_model"

# 결과 파일
SUMMARY_CSV = RESULT_DIR / "model_comparison.csv"


# ============================================================
# 2. 기본 설정
# ============================================================

IMG_SIZE = 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Mac metadata 파일 제외
IMAGE_PATHS = sorted([
    p for p in IMAGE_DIR.glob("*.jpg")
    if not p.name.startswith("._")
])


# ============================================================
# 3. CPU 모니터
# ============================================================

cpu_samples = []
stop_cpu_monitor = threading.Event()


def monitor_cpu():

    while not stop_cpu_monitor.is_set():

        cpu = psutil.cpu_percent(interval=0.2)

        cpu_samples.append(cpu)


# ============================================================
# 4. IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# 5. GT 읽기
# ============================================================

def load_gt():

    gt = {}

    with open(GT_FILE, "r", encoding="utf-8") as f:

        reader = csv.reader(f)

        for row in reader:

            if len(row) < 6:
                continue

            frame = int(row[0])

            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])

            box = [
                x,
                y,
                x + w,
                y + h
            ]

            if frame not in gt:
                gt[frame] = []

            gt[frame].append(box)

    return gt


# ============================================================
# 6. Detection 평가
# ============================================================

def evaluate_detection(model, model_name, gt):

    print()
    print("=" * 70)
    print(f"{model_name} Detection Evaluation")
    print("=" * 70)

    TP = 0
    FP = 0
    FN = 0

    iou_values = []

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 모든 프레임 처리
    # --------------------------------------------------------

    for frame_idx, image_path in enumerate(IMAGE_PATHS, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            classes=[0],
            verbose=False
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:

                predictions.append(box)

        gt_boxes = gt.get(frame_idx, [])

        matched_gt = set()

        # ----------------------------------------------------
        # Prediction → GT 매칭
        # ----------------------------------------------------

        for pred_box in predictions:

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):

                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(
                    pred_box,
                    gt_box
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRESHOLD:

                TP += 1

                matched_gt.add(best_gt_idx)

                iou_values.append(best_iou)

            else:

                FP += 1

        # ----------------------------------------------------
        # 미검출
        # ----------------------------------------------------

        FN += len(gt_boxes) - len(matched_gt)

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    mean_iou = (
        np.mean(iou_values)
        if iou_values
        else 0
    )

    fps = len(IMAGE_PATHS) / elapsed

    print(f"TP          : {TP}")
    print(f"FP          : {FP}")
    print(f"FN          : {FN}")

    print()
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"Mean IoU    : {mean_iou:.4f}")

    print()
    print(f"Total Time  : {elapsed:.2f} sec")
    print(f"FPS         : {fps:.2f}")

    return {
        "model": model_name,
        "fps": fps,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou
    }


# ============================================================
# 7. 성능 Benchmark
# ============================================================

def benchmark_model(model, model_name):

    print()
    print("=" * 70)
    print(f"{model_name} Performance Benchmark")
    print("=" * 70)

    cpu_samples.clear()

    stop_cpu_monitor.clear()

    cpu_thread = threading.Thread(
        target=monitor_cpu,
        daemon=True
    )

    cpu_thread.start()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("Warm-up 진행 중...")

    for image_path in IMAGE_PATHS[:10]:

        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            classes=[0],
            verbose=False
        )

    # --------------------------------------------------------
    # 실제 Benchmark
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for image_path in IMAGE_PATHS:

        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            classes=[0],
            verbose=False
        )

    elapsed = time.perf_counter() - start_time

    stop_cpu_monitor.set()

    cpu_thread.join(timeout=1)

    fps = len(IMAGE_PATHS) / elapsed

    avg_cpu = (
        np.mean(cpu_samples)
        if cpu_samples
        else 0
    )

    max_cpu = (
        np.max(cpu_samples)
        if cpu_samples
        else 0
    )

    avg_ms = (
        elapsed / len(IMAGE_PATHS)
    ) * 1000

    print()
    print(f"Frames      : {len(IMAGE_PATHS)}")
    print(f"Total Time  : {elapsed:.2f} sec")
    print(f"Avg Time    : {avg_ms:.2f} ms/frame")
    print(f"FPS         : {fps:.2f}")
    print(f"Avg CPU     : {avg_cpu:.2f}%")
    print(f"Max CPU     : {max_cpu:.2f}%")

    return {
        "fps": fps,
        "avg_ms": avg_ms,
        "avg_cpu": avg_cpu,
        "max_cpu": max_cpu
    }


# ============================================================
# 8. 모델 크기
# ============================================================

def get_model_size(path):

    path = Path(path)

    if path.is_file():

        return path.stat().st_size / (1024 * 1024)

    if path.is_dir():

        total = 0

        for file in path.rglob("*"):

            if file.is_file():

                total += file.stat().st_size

        return total / (1024 * 1024)

    return 0


# ============================================================
# 9. 결과 저장
# ============================================================

def save_summary(results):

    fieldnames = [
        "model",
        "size_mb",
        "fps",
        "avg_ms",
        "avg_cpu",
        "max_cpu",
        "precision",
        "recall",
        "f1",
        "mean_iou"
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)

    print()
    print(f"비교 결과 저장:")
    print(SUMMARY_CSV)


# ============================================================
# 10. Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VOLLEYBALL MODEL COMPARISON")
    print("=" * 70)

    print()
    print(f"Image Directory : {IMAGE_DIR}")
    print(f"GT File         : {GT_FILE}")
    print(f"Frame Count     : {len(IMAGE_PATHS)}")

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    print()
    print("GT 데이터 로딩...")

    gt = load_gt()

    print(f"GT Frame 수 : {len(gt)}")

    results = []

    # ========================================================
    # 1. PyTorch FP32
    # ========================================================

    print()
    print("\n[1/3] PyTorch FP32")

    model_fp32 = YOLO(PT_MODEL)

    performance = benchmark_model(
        model_fp32,
        "PyTorch FP32"
    )

    detection = evaluate_detection(
        model_fp32,
        "PyTorch FP32",
        gt
    )

    results.append({
        "model": "PyTorch FP32",
        "size_mb": get_model_size(PT_MODEL),
        **performance,
        **{
            k: detection[k]
            for k in [
                "precision",
                "recall",
                "f1",
                "mean_iou"
            ]
        }
    })

    # ========================================================
    # 2. OpenVINO FP32
    # ========================================================

    print()
    print("\n[2/3] OpenVINO FP32")

    model_ov_fp32 = YOLO(
        str(OPENVINO_FP32),
        task="detect"
    )

    performance = benchmark_model(
        model_ov_fp32,
        "OpenVINO FP32"
    )

    detection = evaluate_detection(
        model_ov_fp32,
        "OpenVINO FP32",
        gt
    )

    results.append({
        "model": "OpenVINO FP32",
        "size_mb": get_model_size(OPENVINO_FP32),
        **performance,
        **{
            k: detection[k]
            for k in [
                "precision",
                "recall",
                "f1",
                "mean_iou"
            ]
        }
    })

    # ========================================================
    # 3. OpenVINO INT8
    # ========================================================

    print()
    print("\n[3/3] OpenVINO INT8")

    model_int8 = YOLO(
        str(OPENVINO_INT8),
        task="detect"
    )

    performance = benchmark_model(
        model_int8,
        "OpenVINO INT8"
    )

    detection = evaluate_detection(
        model_int8,
        "OpenVINO INT8",
        gt
    )

    results.append({
        "model": "OpenVINO INT8",
        "size_mb": get_model_size(OPENVINO_INT8),
        **performance,
        **{
            k: detection[k]
            for k in [
                "precision",
                "recall",
                "f1",
                "mean_iou"
            ]
        }
    })

    # ========================================================
    # 최종 결과
    # ========================================================

    save_summary(results)

    print()
    print("=" * 100)
    print("최종 비교 결과")
    print("=" * 100)

    print(
        f"{'Model':<20}"
        f"{'Size(MB)':>10}"
        f"{'FPS':>10}"
        f"{'CPU%':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['model']:<20}"
            f"{r['size_mb']:>10.2f}"
            f"{r['fps']:>10.2f}"
            f"{r['avg_cpu']:>10.2f}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>10.4f}"
            f"{r['f1']:>10.4f}"
            f"{r['mean_iou']:>10.4f}"
        )

    print("-" * 100)

    print()
    print("완료!")
    print(f"결과 CSV: {SUMMARY_CSV}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()