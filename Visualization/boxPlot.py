import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402


def main(dataset_path):
    """Plot per-course grade histograms overlaid by Hogwarts House.

    Reads the dataset and lays out a 4x4 grid of subplots, one per course
    (columns from index 6 onward), hiding any unused slots. For each
    course, overlays box of that course's grades. Its aim is to provide
    an intuition about outliers.The figure is saved to
    "media/Hogwarts_Course_BoxPlots.png" and then displayed.

    Args:
        dataset_path: The path to the CSV dataset file to visualize.

    Returns:
        None. Prints per-course record counts to stdout, saves the
        figure to disk, and displays it in a window.
    """
    print("Box Plot Visualization")
    df = pd.read_csv(dataset_path)

    houses = df['Hogwarts House'].dropna().unique()
    courses = df.columns[6:]  # Assuming the first 6 columns are not courses

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.canvas.manager.set_window_title('Hogwarts Course Box Plots by Course')
    axes = axes.flatten()  # makes it easy to index axes[0] through axes[15]

    for i, course in enumerate(courses):
        ax = axes[i]
        ax.boxplot(df[course].dropna())
        df["z_score"] = (df[course] - df[course].mean()) / df[course].std()
        outliers = len(df[df["z_score"].abs() > 3].index)
        ax.set_title(f'{course} - Outliers: {outliers}', fontsize=10)
        ax.grid(axis='y', alpha=0.75)

    # Only show one legend (from the first subplot) to avoid 13
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=len(houses),
               bbox_to_anchor=(0.5, 0.95))

    # Hide any unused subplots (16 slots - 13 courses = 3 empty)
    for j in range(len(courses), len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.293,
                        wspace=0.136,
                        top=0.88,
                        left=0.052,
                        right=0.988)
    plt.savefig('media/Hogwarts_Course_BoxPlots.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]))
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
