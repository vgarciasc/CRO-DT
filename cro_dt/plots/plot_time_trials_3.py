import matplotlib.pyplot as plt
import numpy as np
import pdb
from cycler import cycler

N_DATASETS = 7

def parse_file(filename):
    output = []
    with open("results/" + filename, "r") as f:
        for line in f:
            if line.startswith("Average elapsed time"):
                output.append((float(line.split(" ")[3].strip()), float(line.split(" ")[5].strip())))
    return output

def parse_files(filenames):
    data = [parse_file(f) for f in filenames]
    avgs = np.array([np.resize([avg for avg, std in depth], N_DATASETS) for depth in data])
    stds = np.array([np.resize([std for avg, std in depth], N_DATASETS) for depth in data])
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

    matrix_avgs, matrix_stds = parse_files(["fairest/matrix_depth_2__2023_04_18-11_32_25_summary.txt",
"fairest/matrix_depth_3__2023_04_18-11_33_12_summary.txt",
"fairest/matrix_depth_4__2023_04_18-11_34_58_summary.txt",
"fairest/matrix_depth_5__2023_04_18-11_38_36_summary.txt",
"fairest/matrix_depth_6__2023_04_18-11_45_15_summary.txt",
"fairest/matrix_depth_7__2023_04_18-11_58_37_summary.txt",
])
    cytree_avgs, cytree_stds = parse_files(["fairest/cytree_depth_2__2023_04_17-23_56_37_summary.txt",
"fairest/cytree_depth_3__2023_04_18-00_00_29_summary.txt",
"fairest/cytree_depth_4__2023_04_18-00_04_25_summary.txt",
"fairest/cytree_depth_5__2023_04_18-00_08_26_summary.txt",
"fairest/cytree_depth_6__2023_04_18-00_12_34_summary.txt",
"fairest/cytree_depth_7__2023_04_18-00_16_57_summary.txt",])
    cupy_avgs, cupy_stds = parse_files(["cupy_depth_2__2023_04_14-02_36_17_summary.txt",
                                        "cupy_depth_3__2023_04_14-02_39_12_summary.txt",
                                        "cupy_depth_4__2023_04_14-02_43_23_summary.txt",
                                        "cupy_depth_5__2023_04_14-02_50_10_summary.txt",
                                        "cupy_depth_6__2023_04_14-03_04_09_summary.txt",
                                        "cupy_depth_7__2023_04_14-03_30_09_summary.txt"])
    tfgpu_avgs, tfgpu_stds = parse_files(["tftotal/tftgpu_2noretrac_depth_2__2023_04_18-10_09_04_summary.txt",
                                        "tftotal/tftgpu_2noretrac_depth_3__2023_04_18-10_09_34_summary.txt",
                                        "tftotal/tftgpu_2noretrac_depth_4__2023_04_18-10_10_11_summary.txt",
                                        "tftotal/tftgpu_2noretrac_depth_5__2023_04_18-10_11_05_summary.txt",
                                        "tftotal/tftgpu_2noretrac_depth_6__2023_04_18-10_12_30_summary.txt",
                                        "tftotal/tftgpu_2noretrac_depth_7__2023_04_18-10_13_23_summary.txt",])
    tfcpu_avgs, tfcpu_stds = parse_files(["conda_vinicius/cv_tensorflowgpu_depth_2__2023_04_16-11_31_27_summary.txt",
                                        "conda_vinicius/cv_tensorflowgpu_depth_3__2023_04_16-11_32_13_summary.txt",
                                        "conda_vinicius/cv_tensorflowgpu_depth_4__2023_04_16-11_33_03_summary.txt",
                                        "conda_vinicius/cv_tensorflowgpu_depth_5__2023_04_16-11_34_02_summary.txt",
                                        "conda_vinicius/cv_tensorflowgpu_depth_6__2023_04_16-11_35_16_summary.txt",
                                        "conda_vinicius/cv_tensorflowgpu_depth_7__2023_04_16-11_37_17_summary.txt",])
    numba_avgs, numba_stds = parse_files(["conda_vinicius/cv_numba_depth_2__2023_04_16-11_41_33_summary.txt",
                                        "conda_vinicius/cv_numba_depth_3__2023_04_16-11_47_11_summary.txt",
                                        "conda_vinicius/cv_numba_depth_4__2023_04_16-11_57_25_summary.txt",
                                        "conda_vinicius/cv_numba_depth_5__2023_04_16-12_17_19_summary.txt",
                                        "conda_vinicius/cv_numba_depth_6__2023_04_16-12_57_32_summary.txt",
                                        "conda_vinicius/cv_numba_depth_7__2023_04_16-14_49_14_summary.txt",])
    tree_avgs, tree_stds = parse_files(["conda_vinicius/cv_tree_depth_2__2023_04_16-18_31_15_summary.txt",
                                        "conda_vinicius/cv_tree_depth_3__2023_04_16-18_48_50_summary.txt",
                                        "conda_vinicius/cv_tree_depth_4__2023_04_16-19_15_36_summary.txt",
                                        "conda_vinicius/cv_tree_depth_5__2023_04_16-19_49_27_summary.txt",
                                        "conda_vinicius/cv_tree_depth_6__2023_04_16-20_29_35_summary.txt",
                                        "conda_vinicius/cv_tree_depth_7__2023_04_16-21_20_06_summary.txt",])

    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(12, 6))
    axs[-1, -1].axis('off')
    for i, dataset in enumerate(datasets):
        x = range(2, 8)
        ax = axs[i//4, i%4]
        ax.plot(range(2, len(cytree_avgs[i]) + 2), cytree_avgs[i], marker="o", label="Cython tree", color="red")
        ax.fill_between(range(2, len(cytree_avgs[i]) + 2), cytree_avgs[i] - cytree_stds[i], cytree_avgs[i] + cytree_stds[i], color="red", alpha=0.2)
        # ax.plot(range(2, len(cupy_avgs[i]) + 2), cupy_avgs[i], marker="s", label="CuPy", color="green")
        # ax.fill_between(range(2, len(cupy_avgs[i]) + 2), cupy_avgs[i] - cupy_stds[i], cupy_avgs[i] + cupy_stds[i], color="green", alpha=0.2)
        ax.plot(range(2, len(matrix_avgs[i]) + 2), matrix_avgs[i], marker="*", label="Proposed matrix encoding", color="blue")
        ax.fill_between(range(2, len(matrix_avgs[i]) + 2), matrix_avgs[i] - matrix_stds[i], matrix_avgs[i] + matrix_stds[i], color="blue", alpha=0.2)
        # ax.plot(range(2, len(numba_avgs[i]) + 2), numba_avgs[i], marker="^", label="Numba", color="orange")
        # ax.fill_between(range(2, len(numba_avgs[i]) + 2), numba_avgs[i] - numba_stds[i], numba_avgs[i] + numba_stds[i], color="orange", alpha=0.2)
        # ax.plot(range(2, len(tree_avgs[i]) + 2), tree_avgs[i], marker="x", label="Scikit-learn tree", color="purple")
        # ax.fill_between(range(2, len(tree_avgs[i]) + 2), tree_avgs[i] - tree_stds[i], tree_avgs[i] + tree_stds[i], color="purple", alpha=0.2)
        ax.plot(range(2, len(tfgpu_avgs[i]) + 2), tfgpu_avgs[i], marker="v", label="TensorFlow GPU", color="cyan")
        ax.fill_between(range(2, len(tfgpu_avgs[i]) + 2), tfgpu_avgs[i] - tfgpu_stds[i], tfgpu_avgs[i] + tfgpu_stds[i], color="cyan", alpha=0.2)
        ax.plot(range(2, len(tfcpu_avgs[i]) + 2), tfcpu_avgs[i], marker="d", label="TensorFlow CPU", color="magenta")
        ax.fill_between(range(2, len(tfcpu_avgs[i]) + 2), tfcpu_avgs[i] - tfcpu_stds[i], tfcpu_avgs[i] + tfcpu_stds[i], color="magenta", alpha=0.2)
        ax.set_title(dataset)
        ax.set_yscale("log")
        
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