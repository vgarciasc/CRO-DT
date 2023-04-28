# CRO-DT

This repository contains the Python implementation of *CRO-DT*, an evolutionary algorithm to train decision trees. The algorithm is built upon the Coral Reef Optimization (CRO) algorithm, in particular its dynamic probabilistic variant with substrate layers (sometimes called DPCRO-SL). All code for CRO is forked from PyCRO-SL.

## How to use it

In the parent folder, run `python -m cro_dt.cro_dt` with the following parameters:

- `-i` or `--dataset`: String. Specifies the dataset to use. All viable datasets and their codes are specified in `sup_configs.py`.
- `-c` or `--cro_config`: String. Should be the name of the configuration file to be used. These files determine the number of generations, population size, available substrates, etc. Some examples are provided in the folder `configs/`. 
- `-s` or `--simulations`: Integer. Specifies how many simulations should be run for each dataset.
- `-d` or `--depth`: Integer. Specifies the depth of the decision trees to evolve. This algorithm only handles complete trees with the specified depth.

Concerning optional parameters:

- `--should_cart_init`: True or False. Determines if the initial population should be started with CART trees. This is what separates CRO-DT from CRO-DT (CS), as described in the paper.
- `--evaluation_scheme`: String. Accepts the following values:  
  - `dx`: the matrix encoding proposed in the paper.  
  - `tree`: the traditional tree encoding.
- `--output_prefix`: String. Defines the prefix appended to the output filename.
- `--should_save_reports`: True or False. Each simulation, PyCRO-SL generates a pyplot figure with information from the run. If this parameters is true, they will all be saved to folder `results/`.
- `--start_from`: Integer. Determines if it should pick up from simulation X, instead of beginning from 0. Useful when you want to write a paper but people keep restarting the machine and you don't want to start over from scratch.
- `--univariate`: True or False. Determines if the trained trees will be univariate or multivariate.

There are more parameters, but they are all optional and involve things that were tested with during our investigation. You may safely ignore them.

For comparing the running times between different fitness evaluation schemes, please use the `measure_time_perf.py` script. It removes excess computations to properly compute the running time of each scheme.