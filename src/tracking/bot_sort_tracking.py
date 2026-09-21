from pathlib import Path
import cv2
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from ultralytics import YOLO


# =========================================================
# 1. 경로 설정
# =========================================================

IMAGE_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\data\sportmot\val\val\v_9MHDmAMxO5I_c009\img1"
)

OUTPUT_CSV = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\results\trajectory.csv"
)

OUTPUT_TRAJECTORY = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\results\trajectory_plot.png"
)

MODEL_PATH = "yolo26s.pt"


# =========================================================
# 2. 모델 로드
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# 3. 이미지 목록
#    ._ 파일은 제외
# =========================================================

image_paths = sorted([
    p for p in IMAGE_DIR.glob("*.jpg")
    if not p.name.startswith("._")
])

print(f"총 프레임 수: {len(image_paths)}")


# =========================================================
# 4. 결과 폴더 생성
# =========================================================

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# 5. 선수별 이동 궤적 저장
#
# trajectories[ID] = [(x1, y1), (x2, y2), ...]
# =========================================================

trajectories = defaultdict(list)


# =========================================================
# 6. CSV 저장 + 추적 시작
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "confidence"
    ])


    # -----------------------------------------------------
    # 프레임 반복
    # -----------------------------------------------------

    for frame_idx, image_path in enumerate(
        image_paths,
        start=1
    ):

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"이미지 읽기 실패: {image_path}")
            continue


        # -------------------------------------------------
        # YOLO + BoT-SORT
        # -------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=0.4,
            verbose=False
        )

        result = results[0]


        # -------------------------------------------------
        # Tracking 결과가 존재하는 경우
        # -------------------------------------------------

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            ids = result.boxes.id.int().cpu().numpy()

            confs = result.boxes.conf.cpu().numpy()


            # ---------------------------------------------
            # 검출된 선수 하나씩 처리
            # ---------------------------------------------

            for box, track_id, confidence in zip(
                boxes,
                ids,
                confs
            ):

                x1, y1, x2, y2 = box


                # -----------------------------------------
                # 중심점 계산
                # -----------------------------------------

                center_x = (x1 + x2) / 2

                center_y = (y1 + y2) / 2


                # -----------------------------------------
                # CSV 저장
                # -----------------------------------------

                writer.writerow([
                    frame_idx,
                    int(track_id),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(center_x),
                    float(center_y),
                    float(confidence)
                ])


                # -----------------------------------------
                # 해당 ID의 궤적에 현재 위치 추가
                # -----------------------------------------

                trajectories[int(track_id)].append(
                    (center_x, center_y)
                )


                # -----------------------------------------
                # Bounding Box
                # -----------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )


                # -----------------------------------------
                # ID 표시
                # -----------------------------------------

                cv2.putText(
                    frame,
                    f"ID: {int(track_id)}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # -----------------------------------------
                # 현재 중심점
                # -----------------------------------------

                cv2.circle(
                    frame,
                    (int(center_x), int(center_y)),
                    5,
                    (0, 0, 255),
                    -1
                )


        # =================================================
        # 이동 궤적 그리기
        # =================================================

        for track_id, points in trajectories.items():

            # 너무 많은 점을 매번 그리지 않도록
            # 최근 궤적을 사용
            recent_points = points[-25:]


            for i in range(1, len(recent_points)):

                pt1 = (
                    int(recent_points[i - 1][0]),
                    int(recent_points[i - 1][1])
                )

                pt2 = (
                    int(recent_points[i][0]),
                    int(recent_points[i][1])
                )


                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    (255, 0, 0),
                    2
                )


        # =================================================
        # 현재 프레임 번호
        # =================================================

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # =================================================
        # 화면 출력
        # 약 25 FPS
        # =================================================

        cv2.imshow(
            "Volleyball - YOLO26 + BoT-SORT + Trajectory",
            frame
        )


        # 25 FPS 정도로 보기 좋게 재생
        if cv2.waitKey(40) & 0xFF == ord("q"):
            break


# =========================================================
# 7. OpenCV 창 종료
# =========================================================

cv2.destroyAllWindows()


print("\nTracking 완료")
print(f"CSV 저장: {OUTPUT_CSV}")


# =========================================================
# 8. 전체 이동 궤적 그래프 생성
# =========================================================

plt.figure(figsize=(12, 7))


for track_id, points in trajectories.items():

    if len(points) < 2:
        continue


    x = [p[0] for p in points]

    y = [p[1] for p in points]


    plt.plot(
        x,
        y,
        linewidth=2,
        label=f"ID {track_id}"
    )


    # 시작점
    plt.scatter(
        x[0],
        y[0],
        s=40
    )


    # 마지막점
    plt.scatter(
        x[-1],
        y[-1],
        s=40
    )


# ---------------------------------------------------------
# 이미지 좌표이므로 Y축을 뒤집음
# ---------------------------------------------------------

plt.gca().invert_yaxis()


plt.xlabel("Center X (pixel)")
plt.ylabel("Center Y (pixel)")

plt.title(
    "Player Movement Trajectories - YOLO26 + BoT-SORT"
)

plt.grid(True)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()


# =========================================================
# 9. 그래프 저장
# =========================================================

plt.savefig(
    OUTPUT_TRAJECTORY,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print(f"궤적 그래프 저장: {OUTPUT_TRAJECTORY}")