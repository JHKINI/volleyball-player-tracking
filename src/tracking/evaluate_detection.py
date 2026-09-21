from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 경로
# ============================================================

IMAGE_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\data\sportmot\val\val\v_9MHDmAMxO5I_c009\img1"
)

GT_PATH = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\data\sportmot\val\val\v_9MHDmAMxO5I_c009\gt\gt.txt"
)

MODEL_PATH = "yolo26s.pt"

OUTPUT_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\results"
)

PREDICTION_CSV = OUTPUT_DIR / "detection_predictions.csv"


# ============================================================
# 설정
# ============================================================

IMG_SIZE = 640

# Precision / Recall / F1 계산용
CONF_THRESHOLD = 0.4

# AP 계산용
# AP는 낮은 confidence 예측까지 포함해서 평가
AP_CONF_THRESHOLD = 0.001

# IoU threshold
MATCH_IOU_THRESHOLD = 0.5


# ============================================================
# IoU 계산
# ============================================================

def calculate_iou(box1, box2):
    """
    box format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )

    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GT 읽기
# ============================================================

def load_gt(gt_path):
    """
    MOT format:

    frame, id, x, y, width, height, ...
    """

    gt = {}

    with open(gt_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split(",")

            frame_id = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            width = float(values[4])
            height = float(values[5])

            box = [
                x,
                y,
                x + width,
                y + height
            ]

            if frame_id not in gt:
                gt[frame_id] = []

            gt[frame_id].append({
                "id": track_id,
                "box": box
            })

    return gt


# ============================================================
# Detection 평가
# ============================================================

def evaluate_at_iou(predictions, gt, iou_threshold):

    tp = 0
    fp = 0
    fn = 0

    matched_ious = []

    for frame_id in sorted(gt.keys()):

        gt_boxes = gt.get(frame_id, [])

        pred_boxes = predictions.get(frame_id, [])

        matched_gt = set()

        # confidence 높은 순서
        pred_boxes = sorted(
            pred_boxes,
            key=lambda x: x["confidence"],
            reverse=True
        )

        for pred in pred_boxes:

            best_iou = 0.0
            best_gt_index = -1

            for gt_index, gt_item in enumerate(gt_boxes):

                if gt_index in matched_gt:
                    continue

                iou = calculate_iou(
                    pred["box"],
                    gt_item["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_index = gt_index

            if best_iou >= iou_threshold:

                tp += 1

                matched_gt.add(best_gt_index)

                matched_ious.append(best_iou)

            else:

                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    mean_iou = (
        np.mean(matched_ious)
        if matched_ious
        else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou
    }


# ============================================================
# AP 계산
# ============================================================

def calculate_ap(predictions, gt, iou_threshold):

    all_predictions = []

    total_gt = 0

    for frame_id in gt:

        total_gt += len(gt[frame_id])

    # 모든 prediction을 하나로 합침
    for frame_id, frame_predictions in predictions.items():

        for pred in frame_predictions:

            all_predictions.append({
                "frame": frame_id,
                "box": pred["box"],
                "confidence": pred["confidence"]
            })

    # confidence 높은 순서
    all_predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    matched_gt = {
        frame_id: set()
        for frame_id in gt
    }

    tp_list = []
    fp_list = []

    for pred in all_predictions:

        frame_id = pred["frame"]

        gt_boxes = gt.get(frame_id, [])

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_item in enumerate(gt_boxes):

            if gt_index in matched_gt[frame_id]:
                continue

            iou = calculate_iou(
                pred["box"],
                gt_item["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= iou_threshold:

            tp_list.append(1)
            fp_list.append(0)

            matched_gt[frame_id].add(best_gt_index)

        else:

            tp_list.append(0)
            fp_list.append(1)

    if total_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    recalls = tp_cumsum / total_gt

    precisions = tp_cumsum / (
        tp_cumsum + fp_cumsum + 1e-12
    )

    # Precision envelope
    precisions = np.maximum.accumulate(
        precisions[::-1]
    )[::-1]

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)

    precision_points = np.zeros_like(
        recall_points
    )

    for i, recall_point in enumerate(recall_points):

        valid = recalls >= recall_point

        if np.any(valid):

            precision_points[i] = np.max(
                precisions[valid]
            )

    ap = np.mean(precision_points)

    return float(ap)


# ============================================================
# 시작
# ============================================================

print("=" * 60)
print("YOLO26s Detection Evaluation")
print("=" * 60)

print(f"GT 경로      : {GT_PATH}")
print(f"이미지 경로   : {IMAGE_DIR}")
print(f"모델         : {MODEL_PATH}")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GT 로드
# ============================================================

print()
print("GT 로딩 중...")

gt = load_gt(GT_PATH)

print(f"GT 프레임 수 : {len(gt)}")
print(
    f"GT 객체 수   : {sum(len(v) for v in gt.values())}"
)


# ============================================================
# 이미지 로드
# ============================================================

image_paths = sorted([
    p for p in IMAGE_DIR.glob("*.jpg")
    if not p.name.startswith("._")
])

print(f"이미지 수     : {len(image_paths)}")


# ============================================================
# 모델 로드
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# Prediction
# ============================================================

predictions = {}

csv_rows = []

print()
print("Detection 시작...")

for frame_index, image_path in enumerate(
    image_paths,
    start=1
):

    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=AP_CONF_THRESHOLD,
        classes=[0],
        verbose=False
    )

    result = results[0]

    predictions[frame_index] = []

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, confidence in zip(
        boxes,
        confs
    ):

        x1, y1, x2, y2 = box

        prediction = {
            "box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ],
            "confidence": float(confidence)
        }

        predictions[frame_index].append(
            prediction
        )

        csv_rows.append([
            frame_index,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(confidence)
        ])

    if frame_index % 50 == 0:
        print(
            f"진행: {frame_index}/{len(image_paths)}"
        )


# ============================================================
# Prediction CSV 저장
# ============================================================

with open(
    PREDICTION_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence"
    ])

    writer.writerows(csv_rows)


print()
print(f"Prediction 저장 완료:")
print(PREDICTION_CSV)


# ============================================================
# Precision / Recall / F1 / IoU
# ============================================================

print()
print("=" * 60)
print("Detection Metrics @ IoU 0.50")
print("=" * 60)

# conf=0.4 적용
filtered_predictions = {}

for frame_id, frame_predictions in predictions.items():

    filtered_predictions[frame_id] = [
        p for p in frame_predictions
        if p["confidence"] >= CONF_THRESHOLD
    ]


metrics = evaluate_at_iou(
    filtered_predictions,
    gt,
    MATCH_IOU_THRESHOLD
)


print(f"TP              : {metrics['TP']}")
print(f"FP              : {metrics['FP']}")
print(f"FN              : {metrics['FN']}")
print(f"Precision       : {metrics['precision']:.4f}")
print(f"Recall          : {metrics['recall']:.4f}")
print(f"F1-score        : {metrics['f1']:.4f}")
print(f"Mean IoU        : {metrics['mean_iou']:.4f}")


# ============================================================
# AP
# ============================================================

print()
print("=" * 60)
print("AP / mAP")
print("=" * 60)

ap50 = calculate_ap(
    predictions,
    gt,
    0.50
)

ap75 = calculate_ap(
    predictions,
    gt,
    0.75
)

ap_values = []

iou_thresholds = np.arange(
    0.50,
    0.96,
    0.05
)

for iou_threshold in iou_thresholds:

    ap = calculate_ap(
        predictions,
        gt,
        float(iou_threshold)
    )

    ap_values.append(ap)

    print(
        f"AP@{iou_threshold:.2f} : {ap:.4f}"
    )

map50_95 = float(
    np.mean(ap_values)
)


# ============================================================
# 최종 결과
# ============================================================

print()
print("=" * 60)
print("FINAL DETECTION RESULTS")
print("=" * 60)

print(f"Precision       : {metrics['precision']:.4f}")
print(f"Recall          : {metrics['recall']:.4f}")
print(f"F1-score        : {metrics['f1']:.4f}")
print(f"Mean IoU        : {metrics['mean_iou']:.4f}")
print(f"AP50            : {ap50:.4f}")
print(f"AP75            : {ap75:.4f}")
print(f"mAP50-95        : {map50_95:.4f}")

print("=" * 60)