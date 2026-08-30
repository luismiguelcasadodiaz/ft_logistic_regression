import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402


def main(dataset_path):
    print("Histogram Visualization")
    df = pd.read_csv(dataset_path)

    houses = df['Hogwarts House'].dropna().unique()
    courses = df.columns[6:]  # Assuming the first 6 columns are not courses

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.canvas.manager.set_window_title('Hogwarts Course Histograms by House')
    axes = axes.flatten()  # makes it easy to index axes[0] through axes[15]

    for i, course in enumerate(courses):
        ax = axes[i]
        records=0
        for house in houses:
            house_data = df[df['Hogwarts House'] == house][course].dropna()

            ax.hist(house_data, bins=30, alpha=0.5, label=house)
            records += len(house_data)
        print(f"Course: {course}, Data Count: {records}")
        ax.set_title(f'{course}', fontsize=10)
        # ax.set_xlabel(course, fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel('Frequency', fontsize=8)
        else:
            ax.set_ylabel('')  # Only label y-axis for the first column
        ax.set_ylim(0, 120)  # <-- caps y-axis at 120
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
                        left=0.03,
                        right=0.988)
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
