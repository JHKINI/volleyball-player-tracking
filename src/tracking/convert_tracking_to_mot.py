from pathlib import Path
import csv


# =========================
# 경로 설정
# =========================
PROJECT_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking"
)

INPUT_CSV = PROJECT_DIR / "results" / "trajectory.csv"

OUTPUT_DIR = PROJECT_DIR / "results" / "mot"
OUTPUT_FILE = OUTPUT_DIR / "v_9MHDmAMxO5I_c009.txt"


# =========================
# 폴더 생성
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CSV → MOT 형식 변환
# =========================
with open(INPUT_CSV, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        for row in reader:
            frame = int(row["frame"])
            track_id = int(row["track_id"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            confidence = float(row["confidence"])

            # xyxy → xywh
            width = x2 - x1
            height = y2 - y1

            # MOTChallenge 10-column format
            writer.writerow([
                frame,
                track_id,
                x1,
                y1,
                width,
                height,
                confidence,
                -1,
                -1,
                -1
            ])


print("=" * 60)
print("BoT-SORT → MOT 형식 변환 완료")
print("=" * 60)
print(f"입력 : {INPUT_CSV}")
print(f"출력 : {OUTPUT_FILE}")
print("=" * 60)