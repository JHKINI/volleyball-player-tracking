import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "../../results/trajectory.csv"


df = pd.read_csv(CSV_PATH)


# ==========================================
# ID별 궤적
# ==========================================

plt.figure(figsize=(12, 7))


for track_id, group in df.groupby("track_id"):

    plt.plot(
        group["center_x"],
        group["center_y"],
        label=f"ID {track_id}"
    )


plt.gca().invert_yaxis()

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Player Trajectory")

plt.legend()

plt.show()