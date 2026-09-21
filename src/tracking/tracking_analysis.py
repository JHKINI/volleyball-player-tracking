import pandas as pd


CSV_PATH = "../../results/trajectory.csv"


df = pd.read_csv(CSV_PATH)


# ID별 등장 프레임
for track_id, group in df.groupby("track_id"):

    frames = group["frame"].sort_values().tolist()

    gaps = []

    for i in range(1, len(frames)):

        gap = frames[i] - frames[i - 1]

        if gap > 1:
            gaps.append(gap)


    print(
        f"ID {track_id}: "
        f"등장 프레임={len(frames)}, "
        f"검출 공백={gaps}"
    )