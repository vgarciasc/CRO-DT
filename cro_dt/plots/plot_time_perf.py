import numpy as np
import matplotlib.pyplot as plt
import json

def plot_time_perf(time_measures):
    n_datasets = len(time_measures)
    width = n_datasets // 2 if n_datasets > 4 else n_datasets
    height = n_datasets % 2 if n_datasets > 4 else 1

    fig, axs = plt.subplots(height, width, sharex=True, figsize=(12, 6))
    if height > 1:
        axs[-1, -1].axis('off')

    dataset_idx = -1
    for (dataset, data) in time_measures.items():
        dataset_idx += 1

        if height > 1:
            ax = axs[dataset_idx // width, dataset_idx % width]
        else:
            ax = axs[dataset_idx]

        get_data = lambda algo : [data[f'depth_{d}'][algo] for d in range(2, 2 + len(data))]
        get_avg_and_std = lambda algo : (np.mean(get_data(algo), axis=1), np.std(get_data(algo), axis=1))
        algorithms = data['depth_2'].keys()

        if 'cytree' in algorithms:
            cytree_avgs, cytree_stds = get_avg_and_std('cytree')
            ax.plot(range(2, len(cytree_avgs) + 2), cytree_avgs, marker="o", label="Traditional encoding (Cython)", color="red")
            ax.fill_between(range(2, len(cytree_avgs) + 2), cytree_avgs - cytree_stds, cytree_avgs + cytree_stds,
                            color="red", alpha=0.2)

        if 'tree' in algorithms:
            tree_avgs, tree_stds = get_avg_and_std('tree')
            ax.plot(range(2, len(tree_avgs) + 2), tree_avgs, marker="x", label="Traditional encoding (Python)", color="purple")
            ax.fill_between(range(2, len(tree_avgs) + 2), tree_avgs - tree_stds, tree_avgs + tree_stds, color="purple", alpha=0.2)

        if 'matrix' in algorithms:
            matrix_avgs, matrix_stds = get_avg_and_std('matrix')
            ax.plot(range(2, len(matrix_avgs) + 2), matrix_avgs, marker="*", label="Proposed encoding (NumPy)", color="blue")
            ax.fill_between(range(2, len(matrix_avgs) + 2), matrix_avgs - matrix_stds, matrix_avgs + matrix_stds, color="blue", alpha=0.2)

        if 'tf' in algorithms:
            tf_avgs, tf_stds = get_avg_and_std('tf')
            ax.plot(range(2, len(tf_avgs) + 2), tf_avgs, marker="s", label="Proposed encoding (Tensorflow)", color="green")
            ax.fill_between(range(2, len(tf_avgs) + 2), tf_avgs - tf_stds, tf_avgs + tf_stds, color="green", alpha=0.2)

        if 'tf_batch' in algorithms:
            tf_batch_avgs, tf_batch_stds = get_avg_and_std('tf_batch')
            ax.plot(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs, marker="s", label="Proposed encoding (Tensorflow batch)", color="cyan")
            ax.fill_between(range(2, len(tf_batch_avgs) + 2), tf_batch_avgs - tf_batch_stds, tf_batch_avgs + tf_batch_stds, color="cyan", alpha=0.2)

        ax.set_title(dataset)
        ax.set_yscale("log")

        if dataset_idx >= 1 * 4 or dataset_idx == 3:
            ax.set_xlabel("Depth")

        if dataset_idx == n_datasets - 1:
            # ax.legend(bbox_to_anchor=(1.2, 0.5), loc="center left")
            ax.legend()

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
    with open("results/time_measures.json", "r") as f:
        time_measures = json.load(f)

    plot_time_perf(time_measures)
