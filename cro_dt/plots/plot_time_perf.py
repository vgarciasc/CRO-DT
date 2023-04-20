import numpy as np
import matplotlib.pyplot as plt
import math
import json

def plot_time_perf(time_measures):
    transposed_time_measures = {}
    for algo, datasets in time_measures.items():
        for dataset, configs in datasets.items():
            if dataset not in transposed_time_measures:
                transposed_time_measures[dataset] = {}
            for config, value in configs.items():
                if config not in transposed_time_measures[dataset]:
                    transposed_time_measures[dataset][config] = {}
                transposed_time_measures[dataset][config][algo] = value

    time_measures = transposed_time_measures

    n_datasets = len(time_measures)
    width = 4 if n_datasets > 4 else n_datasets
    height = math.ceil(n_datasets / 4) if n_datasets > 4 else 1

    fig, axs = plt.subplots(height, width, figsize=(12, 6))
    if height > 1:
        axs[-1, -1].axis('off')

    dataset_idx = -1
    for (dataset, data) in time_measures.items():
        dataset_idx += 1

        if height > 1:
            ax = axs[dataset_idx // width, dataset_idx % width]
        else:
            ax = axs[dataset_idx]

        get_avg = lambda algo : np.array([np.mean(data[f'depth_{d}'][algo]) for d in range(2, 2 + len(data)) if algo in data[f'depth_{d}'].keys()])
        get_std = lambda algo : np.array([np.std(data[f'depth_{d}'][algo]) for d in range(2, 2 + len(data)) if algo in data[f'depth_{d}'].keys()])
        algorithms = data['depth_2'].keys()

        if 'tree' in algorithms:
            tree_avgs, tree_stds = get_avg('tree'), get_std('tree')
            ax.plot(range(2, len(tree_avgs) + 2), tree_avgs, marker="*", label="Traditional encoding (Python)", color="red")
            ax.fill_between(range(2, len(tree_avgs) + 2), tree_avgs - tree_stds, tree_avgs + tree_stds, color="red", alpha=0.2)

        if 'matrix' in algorithms:
            matrix_avgs, matrix_stds = get_avg('matrix'), get_std('matrix')
            ax.plot(range(2, len(matrix_avgs) + 2), matrix_avgs, marker="*", label="Proposed encoding (NumPy)", color="blue")
            ax.fill_between(range(2, len(matrix_avgs) + 2), matrix_avgs - matrix_stds, matrix_avgs + matrix_stds, color="blue", alpha=0.2)

        if 'cytree' in algorithms:
            cytree_avgs, cytree_stds = get_avg('cytree'), get_std('cytree')
            ax.plot(range(2, len(cytree_avgs) + 2), cytree_avgs, marker="o", label="Traditional encoding (C)", color="orange")
            ax.fill_between(range(2, len(cytree_avgs) + 2), cytree_avgs - cytree_stds, cytree_avgs + cytree_stds,
                            color="orange", alpha=0.2)

        # if 'tf' in algorithms:
        #     tf_avgs, tf_stds = get_avg('tf'), get_std('tf')
        #     ax.plot(range(2, len(tf_avgs) + 2), tf_avgs, marker="s", label="Proposed encoding (TF)", color="green")
        #     ax.fill_between(range(2, len(tf_avgs) + 2), tf_avgs - tf_stds, tf_avgs + tf_stds, color="green", alpha=0.2)

        if 'tf_batch' in algorithms:
            tf_batch_avgs, tf_batch_stds = get_avg('tf_batch'), get_std('tf_batch')
            ax.plot(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs, marker="s", label="Proposed encoding (TF batch)", color="cyan")
            ax.fill_between(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs - tf_batch_stds, tf_batch_avgs + tf_batch_stds, color="cyan", alpha=0.2)

        if 'tf_batch_cpu' in algorithms:
            tf_batch_avgs, tf_batch_stds = get_avg('tf_batch_cpu'), get_std('tf_batch_cpu')
            ax.plot(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs, marker="s", label="Proposed encoding (TF batch CPU)", color="magenta")
            ax.fill_between(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs - tf_batch_stds, tf_batch_avgs + tf_batch_stds, color="magenta", alpha=0.2)

        if 'tf_cpu' in algorithms:
            tf_batch_avgs, tf_batch_stds = get_avg('tf_cpu'), get_std('tf_cpu')
            ax.plot(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs, marker="s", label="Proposed encoding (TF CPU)", color="navy")
            ax.fill_between(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs - tf_batch_stds, tf_batch_avgs + tf_batch_stds, color="navy", alpha=0.2)

        ax.set_title(dataset)
        ax.set_yscale("log")

        if dataset_idx >= 1 * 4 or dataset_idx == 3:
            ax.set_xlabel("Depth")

        if dataset_idx == n_datasets - 1:
            ax.legend(bbox_to_anchor=(1.1, 0.5), loc="center left")
            # ax.legend()

    if height > 1:
        axs[0, 0].set_ylabel("Training time (s)")
        axs[1, 0].set_ylabel("Training time (s)")
    else:
        axs[0].set_ylabel("Training time (s)")

    plt.subplots_adjust(top=0.917,
                        bottom=0.137,
                        left=0.066,
                        right=0.986,
                        hspace=0.276,
                        wspace=0.244)
    plt.show()

if __name__ == "__main__":
    with open("results/time_measures_n100g100s10.json", "r") as f:
        time_measures = json.load(f)

    plot_time_perf(time_measures)
