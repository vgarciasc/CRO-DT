import numpy as np
import pdb
from rich import print

if __name__ == "__main__":
    datasets = ["Artificial N=100, C=3, P=2", "Artificial N=1000, C=3, P=2", "Artificial N=1000, C=3, P=10",
        "Artificial N=1000, C=10, P=10", "Artificial N=10000, C=3, P=2", "Artificial N=10000, C=3, P=2",
        "Artificial N=100000, C=10, P=10"]

    filenames = ["matrix_depth2_full.txt", "matrix_depth3_full.txt", "matrix_depth4_full.txt", "matrix_depth5_full.txt",
        "matrix_depth6_full.txt", "matrix_depth7_full.txt", "matrix_depth8_full.txt", "matrix_depth9_full.txt",
        "tree_depth2_full.txt", "tree_depth3_full.txt", "tree_depth4_full.txt", "tree_depth5_full.txt",
        "tree_depth6_full.txt", "tree_depth7_full.txt", "tree_depth8_full.txt", "tree_depth9_full.txt"]
    
    for filename in filenames:
        elapsed_times = []
        with open(f"results/final/2023-01/time_trial/{filename}", "r") as file:
            for line in file.readlines():
                if "DATASET:" in line:
                    elapsed_times.append([])
                
                if "Elapsed time" in line:
                    elapsed_times[-1].append(float(line.split(": ")[-1]))

            print(f"{'-'*50}\nFilename: {filename}\n")
            for i, dataset in enumerate(datasets):
                if len(elapsed_times[i]) < 5:
                    print(f"[red]Less than 5 simulations!![/red]")

                string = ""
                string += f"DATASET: {dataset}\n"
                string += f"  Elapsed time: {'{:.3f}'.format(np.mean(elapsed_times[i][:5]))} ± {'{:.3f}'.format(np.std(elapsed_times[i][:5]))}\n" 
                print(string)