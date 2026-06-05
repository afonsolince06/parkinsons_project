import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "my_abc_grid_search_results.csv"
PLOT_FILE = "abc_grid_search_plot.png"


def main():
    df = pd.read_csv(RESULTS_FILE)

    df = df.sort_values(by="mean_fitness", ascending=False)
    top5 = df.head(5)

    labels = [
        f"colony={int(row['colony_size'])} | iter={int(row['n_iterations'])} | limit={int(row['limit'])}"
        for _, row in top5.iterrows()
    ]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, top5["mean_fitness"])

    plt.xlabel("Mean F1-score")
    plt.title("ABC Grid Search — Top 5 Configurations by Mean F1-score")

    plt.xlim(
        top5["mean_fitness"].min() - 0.01,
        top5["mean_fitness"].max() + 0.01
    )

    plt.gca().invert_yaxis()

    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            ha="left",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot: {PLOT_FILE}")


if __name__ == "__main__":
    main()