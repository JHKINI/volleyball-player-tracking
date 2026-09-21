import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "../../results/trajectory.csv"


df = pd.read_csv(CSV_PATH)


plt.figure(figsize=(12, 7))


plt.hist2d(
    df["center_x"],
    df["center_y"],
    bins=(50, 30)
)


plt.gca().invert_yaxis()

plt.colorbar(
    label="Number of observations"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Player Position Heatmap")

plt.show()