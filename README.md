# Differentiable Bayesian Filters
This repository contains our study on Differentiable Bayesian Filters.

The package provides implementations of both standard and differentiable Particle Filters, as well as
already trained models and a simple simulation environment with moving discs, which should be followed
by the filters based on the beacon measurements.


## Installation
We recommend creating a virtual/conda environment for the installation.
This code was tested on both `python=3.8` and `python=3.9`.

Run the following commands in the repository`s main directory:
```bat
pip install -r requirements.txt
pip install .
```


## Usage
### I. Training
The training script [train.py](./pf/train.py) is used to train the different models.
In the script, hyperparameters, path to datasets and other settings can be easily changed.
The training itself can be executed by running the script:
```bat
python pf/train.py
```

### II. Inference and Testing
The [try.py](./pf/try.py) script can be used to easily test and visualize different
filters, as well as create datasets that can be used for training or validation, by running:
```bat
python pf/try.py
```
> Note: Please check all the parameters before running the filter!

* For creating a dataset, the following snippet should be included with a specified path:
    ````
    if __name__ == "__main__":
        create_dataset("path")
    ````
  Other settings can be also changed, for more information please take a look at [dataset.py](./pf/utils/dataset.py).
  Please also study the contents of the [datasets](./datasets) folder to find suitable created dataset.

* For testing the standard particle filter, the following snippet must be included:
    ````
    if __name__ == "__main__":
        run_filter()
    ````
    Parameters regarding the simulation as well as other hyperparameters can be easily changed in the function.

* For testing the differentiable particle filter, the following snippet must be included:
    ````
    if __name__ == "__main__":
        run_diff_filter()
    ````
    Parameters regarding the simulation, other hyperparameters as well as the model path must be appropriately specified
    in the function.

Before running the script, we advise checking the [saved_models](./saved_models) folder,
which contains already trained models in different environments.

For the saved models, we provide the state dicts with trained weights, the TF-records,
`setup.txt` file with model parameters, as well as some visualization videos:
* [saved_models](./saved_models)
  * [saved_model.pth](./saved_models/saved_model.pth)
  * [exp0](./saved_models/exp0)
    * [saved_model_final_presentation_mse_loss.pth](./saved_models/exp0/saved_model_final_presentation_mse_loss.pth)
  * [exp1](./saved_models/exp1)
    * [saved_model1646457016.757477.pth](./saved_models/exp1/saved_model1646457016.757477.pth)
    * [events.out.tfevents.1646448859.ZIZOS-PC.11788.0](./saved_models/exp1/events.out.tfevents.1646448859.ZIZOS-PC.11788.0)
  * [exp2](./saved_models/exp2)
    * [saved_model1646605119.4737.pth](./saved_models/exp2/saved_model1646605119.4737.pth)
    * [setup.txt](./saved_models/exp2/setup.txt)
  * [exp3](./saved_models/exp3)
    * [saved_model1646628663.257632.pth](./saved_models/exp3/saved_model1646628663.257632.pth)
    * [events.out.tfevents.1646624040.ZIZOS-PC.18063.0](./saved_models/exp3/events.out.tfevents.1646624040.ZIZOS-PC.18063.0)
    * [setup.txt](./saved_models/exp3/setup.txt)
  * [exp5](./saved_models/exp5)
    * [saved_model1646635129.851859.pth](./saved_models/exp5/saved_model1646635129.851859.pth)
    * [events.out.tfevents.1646633540.ZIZOS-PC.20705.0](./saved_models/exp5/events.out.tfevents.1646633540.ZIZOS-PC.20705.0)
    * [output.avi](./saved_models/exp5/output.avi)
    * [output1.avi](./saved_models/exp5/output1.avi)
    * [output2.avi](./saved_models/exp5/output2.avi)
    * [setup.txt](./saved_models/exp5/setup.txt)
  * [exp4](./saved_models/exp4)
    * [saved_model1646632935.890744.pth](./saved_models/exp4/saved_model1646632935.890744.pth)
    * [events.out.tfevents.1646630099.ZIZOS-PC.19699.0](./saved_models/exp4/events.out.tfevents.1646630099.ZIZOS-PC.19699.0)
    * [output.avi](./saved_models/exp4/output.avi)
    * [output1.avi](./saved_models/exp4/output1.avi)
    * [output3.avi](./saved_models/exp4/output3.avi)
    * [setup.txt](./saved_models/exp4/setup.txt)


## Code Structure
The implemented filters and their components can be found under:
* [pf](./pf)
  * [filters](./pf/filters)
    * [particle_filter.py](./pf/filters/particle_filter.py)
    * [diff_particle_filter.py](./pf/filters/diff_particle_filter.py)
  * [models](./pf/models)
    * [observation_model.py](./pf/models/observation_model.py)
    * [motion_model.py](./pf/models/motion_model.py)

Our simulation environment can be found under:
* [simulation](./pf/simulation)
  * [animation.py](./pf/simulation/animation.py)
  * [simulation_env.py](./pf/simulation/simulation_env.py)


## Repository Structure
Currently, there are un-merged branches in the repository:
* `dev-fixed-cov` contains the older model, which can only be learned with fixed covariance matrices.
* `experiment-1st-milestone` contains code to generate the RMSE graphics (from the paper),
which are then used to compare different models between each other.

We are not planning to either merge or delete these branches.
