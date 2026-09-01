import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402



def main(dataset_path):
    print("Scatter Plot Visualization")
    df = pd.read_csv(dataset_path)
    print(f"Data shape after dropping NaN values: {df.shape}")
    features = df.columns
    courses = [ feature for feature in features[6:] if not feature.endswith("_missing")]
    print(courses)
    n_courses = len(courses)
    fig, axes = plt.subplots(n_courses, n_courses, figsize=(20, 16))
    fig.canvas.manager.set_window_title('Hogwarts Courses Scatter plots')
    axes = axes.flatten()  # makes it easy to index axes[0] through axes[n]
    # Define Hogwarts colors
    house_colors = {
    'Gryffindor': '#740001',  # Scarlet
    'Slytherin':  '#1a472a',  # Green
    'Ravenclaw':  '#0e1a40',  # Blue
    'Hufflepuff': '#ecb939'   # Yellow
}
    colors = df["Hogwarts House"].map(house_colors)
    for i in range(n_courses):
        for j in range(n_courses):
            idx = i * n_courses + j
            ax = axes[idx]
            if j <= i:
                ax.scatter(df[courses[j]], df[courses[i]], marker='.', s=1, c=colors)
            if idx % n_courses == 0:
                ax.set_ylabel(courses[i][0:6], fontsize=8)
                ax.tick_params(axis='y', labelsize=6)
                ax.set_yticks([0,1])
            else:
                ax.set_ylabel('')
                ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{courses[j][0:6]}', fontsize=8)
                ax.set_xticks([])   
            elif i != (n_courses - 1):
                ax.set_title('') 
                ax.set_xlabel('')
                ax.set_xticks([])
            else:
                ax.tick_params(axis='x', labelsize=6)
                ax.set_xticks([0,1])   


    plt.subplots_adjust(hspace=0.00,
                        wspace=0.00,
                        top=0.98,
                        bottom=0.02,
                        left=0.019,
                        right=1)
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
