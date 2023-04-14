import matplotlib.pyplot as plt
import numpy as np
import pdb
from cycler import cycler

def parse_file(filename):
    output = []
    with open("results/" + filename, "r") as f:
        for line in f:
            if line.startswith("Average elapsed time"):
                output.append((float(line.split(" ")[3].strip()), float(line.split(" ")[5].strip())))
    return output

def parse_files(filenames):
    data = [parse_file(f) for f in filenames]
    avgs = np.array([[avg for avg, std in depth] for depth in data])
    stds = np.array([[std for avg, std in depth] for depth in data])
    return avgs.T, stds.T

if __name__ == "__main__":
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    datasets = ["Artificial_1\n(N=100, C=3, P=2)",
        "Artificial_2\n(N=1000, C=3, P=2)",
        "Artificial_3\n(N=1000, C=3, P=10)",
        "Artificial_4\n(N=1000, C=10, P=10)",
        "Artificial_5\n(N=10000, C=3, P=2)",
        "Artificial_6\n(N=10000, C=3, P=10)",
        "Artificial_7\n(N=100000, C=10, P=10)"]

    matrix_avgs, matrix_stds = parse_files(["matrix_depth_2__2023_04_14-00_08_12_summary.txt",
                                            "matrix_depth_3__2023_04_14-00_09_49_summary.txt",
                                            "matrix_depth_4__2023_04_14-00_13_13_summary.txt",
                                            "matrix_depth_5__2023_04_14-00_22_54_summary.txt",
                                            "matrix_depth_6__2023_04_14-00_42_15_summary.txt",
                                            "matrix_depth_7__2023_04_14-01_19_51_summary.txt",])
    cytree_avgs, cytree_stds = parse_files(["cytree_depth_2__2023_04_14-13_51_19_summary.txt",
                                            "cytree_depth_3__2023_04_14-13_58_08_summary.txt",
                                            "cytree_depth_4__2023_04_14-14_05_06_summary.txt",
                                            "cytree_depth_5__2023_04_14-14_12_07_summary.txt",
                                            "cytree_depth_6__2023_04_14-14_19_21_summary.txt",
                                            "cytree_depth_7__2023_04_14-14_26_58_summary.txt",])
    cupy_avgs, cupy_stds = parse_files(["cupy_depth_2__2023_04_14-02_36_17_summary.txt",
                                        "cupy_depth_3__2023_04_14-02_39_12_summary.txt",
                                        "cupy_depth_4__2023_04_14-02_43_23_summary.txt",
                                        "cupy_depth_5__2023_04_14-02_50_10_summary.txt",
                                        "cupy_depth_6__2023_04_14-03_04_09_summary.txt",
                                        "cupy_depth_7__2023_04_14-03_30_09_summary.txt"])
    numba_avgs, numba_stds = parse_files(["numba_depth_2__2023_04_14-09_45_54_summary.txt",
                                          "numba_depth_3__2023_04_14-10_06_58_summary.txt",
                                          "numba_depth_4__2023_04_14-10_49_36_summary.txt",
                                          "numba_depth_5__2023_04_14-14_34_54_summary.txt",])
    tree_avgs, tree_stds = parse_files(["single_run/tree_depth_2__2023_04_13-21_37_48_summary.txt"])

    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(12, 6))
    axs[-1, -1].axis('off')
    for i, dataset in enumerate(datasets):
        x = range(2, 8)
        ax = axs[i//4, i%4]
        ax.plot(range(2, len(cytree_avgs[i]) + 2), cytree_avgs[i], marker="o", label="Cython tree", color="red")
        ax.fill_between(range(2, len(cytree_avgs[i]) + 2), cytree_avgs[i] - cytree_stds[i], cytree_avgs[i] + cytree_stds[i], color="red", alpha=0.2)
        ax.plot(range(2, len(cupy_avgs[i]) + 2), cupy_avgs[i], marker="s", label="CuPy", color="green")
        ax.fill_between(range(2, len(cupy_avgs[i]) + 2), cupy_avgs[i] - cupy_stds[i], cupy_avgs[i] + cupy_stds[i], color="green", alpha=0.2)
        ax.plot(range(2, len(matrix_avgs[i]) + 2), matrix_avgs[i], marker="*", label="Proposed matrix encoding", color="blue")
        ax.fill_between(range(2, len(matrix_avgs[i]) + 2), matrix_avgs[i] - matrix_stds[i], matrix_avgs[i] + matrix_stds[i], color="blue", alpha=0.2)
        ax.plot(range(2, len(numba_avgs[i]) + 2), numba_avgs[i], marker="^", label="Numba", color="orange")
        ax.fill_between(range(2, len(numba_avgs[i]) + 2), numba_avgs[i] - numba_stds[i], numba_avgs[i] + numba_stds[i], color="orange", alpha=0.2)
        ax.plot(range(2, len(tree_avgs[i]) + 2), tree_avgs[i], marker="x", label="Scikit-learn tree", color="purple")
        ax.fill_between(range(2, len(tree_avgs[i]) + 2), tree_avgs[i] - tree_stds[i], tree_avgs[i] + tree_stds[i], color="purple", alpha=0.2)
        ax.set_title(dataset)
        # ax.set_yscale("log")
        
        if i >= 1*4 or i == 3:
            ax.set_xlabel("Depth")
    
        if i == 6:
            ax.legend(bbox_to_anchor=(1.2, 0.5), loc="center left")

    axs[0, 0].set_ylabel("Training time (s)")
    axs[1, 0].set_ylabel("Training time (s)")
    plt.subplots_adjust(top=0.917,
        bottom=0.137,
        left=0.066,
        right=0.986,
        hspace=0.276,
        wspace=0.244)
    plt.show()