# tum-adlr-ws21-10
## Differentiable Bayesian Filters
This repository contains our study in Differentiable Bayesian Filters.

## Installation
We recomend creating a virtual/conda environment for Installation. This code was tested using python ```3.8```.

In the repository`s main directory
```bat
pip install -r requirements.txt
pip install .
```

## Usage:
### Training: 
The Training script [train.py](./pf/train.py) is used to train the different models. In the script, hyperparameters, path to datasets and other settings can be easily changed. An example on running the script:
```bat
python pf/train.py
```

### Inference and Testing:
The [try.py](./pf/try.py) script can be used to easily test and infere different filters, as well as create datasets that can be used for training or validation. An example on running the script:
```bat
python pf/try.py
```

* For creating a dataset, the following snippet should be included, path can be specified. Other settings can be also changed, for more information please take a look at [dataset.py](./pf/utils/dataset.py).

    ````
    if __name__ == "__main__":
        create_dataset("path")
    ````

* For testing the standard particle filter, the following snippet must be included:
    ````
    if __name__ == "__main__":
        run_filter()
    ````
    Parameters regarding the simulation and other hyper parameters can be easily changed in the function.

* For testing the Differntiable particle filter, the following snippet must be included:
    ````
    if __name__ == "__main__":
        run_diff_filter()
    ````
    Parameters regarding the simulation and other hyper parameters as well as the model path must be appropriately specified in the function, more information about the models setub can be found in the setup.txt for each experiment,

    Saved models and experiment results can be found under the directory ```saved_models```. We provide the trained models, the ```TF-Records```, as well as some videos for visulaization.

    .
     * [saved_models](./saved_models)
     * [exp1](./saved_models/exp1)
       * [saved_model1646457016.757477.pth](./saved_models/exp1/saved_model1646457016.757477.pth)
       * [events.out.tfevents.1646448859.ZIZOS-PC.11788.0](./saved_models/exp1/events.out.tfevents.1646448859.ZIZOS-PC.11788.0)
     * [saved_model.pth](./saved_models/saved_model.pth)
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
     * [exp2](./saved_models/exp2)
       * [saved_model1646605119.4737.pth](./saved_models/exp2/saved_model1646605119.4737.pth)
       * [setup.txt](./saved_models/exp2/setup.txt)
     * [exp0](./saved_models/exp0)
         * [saved_model_final_presentation_mse_loss.pth](./saved_models/exp0/saved_model_final_presentation_mse_loss.pth)

## Code Structure

### The implemented filters and their components can be found under:

.
 * [pf](./pf)
   * [filters](./pf/filters)
     * [particle_filter.py](./pf/filters/particle_filter.py)
     * [diff_particle_filter.py](./pf/filters/diff_particle_filter.py)
   * [models](./pf/models)
     * [observation_model.py](./pf/models/observation_model.py)
     * [motion_model.py](./pf/models/motion_model.py)

### Our Simulation environment can be found under:
.
 * [simulation](./pf/simulation)
     * [animation.py](./pf/simulation/animation.py)
     * [simulation_env.py](./pf/simulation/simulation_env.py)