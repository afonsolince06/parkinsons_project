import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("abc_grid_search_results.csv")

top5 = df.sort_values(by="f1", ascending=False).head(5)

labels = [
    f"colony={int(row['colony_size'])} | iter={int(row['n_iterations'])} | limit={int(row['limit'])}"
    for _, row in top5.iterrows()
]

plt.figure(figsize=(11, 5.5))
bars = plt.barh(labels, top5["f1"])

plt.xlabel("F1-score")
plt.title("ABC Grid Search — Top 5 Configurations")
plt.xlim(top5["f1"].min() - 0.01, top5["f1"].max() + 0.01)
plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.4f}",
        va="center",
        ha="left",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("abc_grid_search_plot.png", dpi=300, bbox_inches="tight")
plt.show()