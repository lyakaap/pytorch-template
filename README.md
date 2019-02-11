# pytorch-template

This repository is my PyTorch project template (for Kaggle and research).
Currently, this repository supports:
* some loss functions for semantic segmentation (src/losses.py)
* mean IoU cuda implementation (src/metrics.py)
* basic UNet implementation (src/models.py)
* learning rate scheduler (src/lr_scheduler.py)
* useful debugging module (src/debug.py)

## Enviroments
This repository supports PyTorch >= 1.0. You can setup the enviroment using docker/Dockerfile (Ubuntu 16.04, cuda 9.2).

## Run Experiments
Set working directory to "experiments" and run below commands,
```
# run training and evaluation (with saving checkpoints)
python exp0.py job --devices 0,1

# Run grid-search for better hyperparameter set.
# Parameter space can be set in "exp0.py".
# In below setting, each trial is conducted on a single gpu device 
# and thus whole tuning processes are launched on multiple gpu devices in parallel.
python exp0.py tuning --devices 0,1 --n-gpu 1 --mode 'grid'
```
