# SHELLEY
This repository contains the code for SHELLEY, a tool that facilitates the development, testing, and combination of learning-based network alignment algorithms by providing a set of modules that allow for the recreation and combination of both representation learning methods (RLMs) and deep matching methods (DMMs). With SHELLEY is it possible to test the algorithms on real and semi-synthetic data sets.

## Prerequisites
If you use a `conda` you can simply import the virtual environment with:
```
conda env create -f environment.yml
```

Otherwise you can see `requirements.txt` to see the list of required packages.

## How to run
To run one experiment use:
```
python train_eval.py -e experiments/<experiment_name>.py
```

If you want to run all the experiments in `experiments/` directory, you can use the script:
```
./scripts/run_experiments.sh
```
