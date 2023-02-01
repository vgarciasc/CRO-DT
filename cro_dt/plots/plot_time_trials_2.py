import matplotlib.pyplot as plt
import numpy as np
import pdb
from cycler import cycler


if __name__ == "__main__":
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    datasets = ["Artificial_1\n(N=100, C=3, P=2)",
        "Artificial_2\n(N=1000, C=3, P=2)",
        "Artificial_3\n(N=1000, C=3, P=10)",
        "Artificial_4\n(N=1000, C=10, P=10)",
        "Artificial_5\n(N=10000, C=3, P=2)",
        "Artificial_6\n(N=10000, C=3, P=10)",
        "Artificial_7\n(N=100000, C=10, P=10)"]
    
    tree_avgs = np.array([[1.83, 12.58, 12.68, 12.68, 120.38, 120.21, 1192.53], 
        [2.46, 17.57, 17.70, 17.65, 168.73, 169.25, 1689.06], 
        [3.15, 23.02, 23.12, 23.15, 217.95, 217.69, 2175.44], 
        [4.04, 28.43, 28.57, 28.42, 268.17, 267.26, 2683.12], 
        [5.28, 34.00, 34.33, 34.42, 317.50, 318.12, 3165.52], 
        [7.36, 40.71, 41.13, 41.33, 369.63, 368.32, 3675.48], 
        [11.66, 49.80, 50.45, 50.58, 427.75, 428.20, 4249.49], 
        [19.72, 62.77, 64.01, 64.52, 490.83, 487.78, 4770.71]])

    tree_stds = np.array([[0.01, 0.11, 0.05, 0.06, 0.69, 0.61, 2.75], 
        [0.01, 0.07, 0.12, 0.03, 1.23, 0.61, 2.98], 
        [0.01, 0.04, 0.08, 0.07, 0.98, 1.04, 5.42], 
        [0.03, 0.06, 0.07, 0.06, 1.35, 0.83, 8.65], 
        [0.03, 0.06, 0.10, 0.11, 0.32, 1.06, 8.32], 
        [0.01, 0.09, 0.07, 0.06, 0.62, 0.97, 6.52], 
        [0.11, 0.13, 0.06, 0.13, 1.83, 0.51, 11.88], 
        [0.06, 0.12, 0.25, 0.55, 1.44, 3.03, 9.95]])

    matrix_avgs = np.array([[0.46, 0.55, 0.56, 0.57, 1.41, 1.42, 13.22], 
        [0.54, 0.73, 0.76, 0.75, 3.90, 4.05, 27.29], 
        [0.77, 1.17, 1.21, 1.19, 10.62, 9.55, 101.69], 
        [0.97, 3.11, 3.32, 3.32, 24.99, 25.05, 253.08], 
        [1.60, 5.72, 5.83, 5.84, 51.19, 48.90, 541.49], 
        [5.86, 17.79, 18.17, 16.97, 113.10, 107.63, 1417.09], 
        [11.65, 43.63, 44.62, 39.36, 253.95, 227.08, 2836.74], 
        [37.51, 130.53, 170.81, 127.57, 694.26, 703.08, 6779.06]])
    
    matrix_stds = np.array([[0.01, 0.00, 0.01, 0.00, 0.01, 0.01, 0.13], 
        [0.00, 0.01, 0.00, 0.00, 0.24, 0.01, 0.17], 
        [0.01, 0.01, 0.01, 0.01, 3.03, 2.54, 22.27], 
        [0.00, 0.22, 0.06, 0.04, 2.01, 1.96, 62.10], 
        [0.00, 0.08, 0.04, 0.10, 5.08, 4.89, 134.29], 
        [0.11, 5.36, 7.27, 7.03, 26.13, 29.04, 10.43], 
        [0.08, 12.03, 11.68, 12.11, 65.40, 65.69, 5.74], 
        [3.00, 31.25, 69.44, 46.15, 115.14, 117.83, 18.15]])

    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(12, 6))
    axs[-1, -1].axis('off')
    for i, dataset in enumerate(datasets):
        x = range(2, 10)
        ax = axs[i//4, i%4]
        ax.plot(x, tree_avgs[:,i], marker="*", label="Traditional tree encoding", color="red")
        ax.fill_between(x, tree_avgs[:,i] - tree_stds[:,i], tree_avgs[:,i] + tree_stds[:,i], color="red", alpha=0.2)
        ax.plot(x, matrix_avgs[:,i], marker="*", label="Proposed matrix encoding", color="blue")
        ax.fill_between(x, matrix_avgs[:,i] - matrix_stds[:,i], matrix_avgs[:,i] + matrix_stds[:,i], color="blue", alpha=0.2)
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