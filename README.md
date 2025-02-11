# Inundation-emulator

Site specific emulator for tsunami run-up simulations. This repository contains a python implementation using [Tensorflow](https://www.tensorflow.org/) of the neural network developped in [Machine learning emulation of high resolution inundation maps](https://doi.org/10.1093/gji/ggae151).

The application of the code has to be carried out in two steps:

1. (Preparation - `create_emulator.py`) First the model has to be trained for the specific site. To this end a suitably selected training set of sufficient size is needed.

1. (Operation - `predict.py`) The trained model may be applied for predictions.

The main functionality of the code is implemented in `data_reader.py` and `emulator.py`. The training proceedure generates a set of plots and logs to visualize whether training was successful or not. Furthermore generation of a mask (based on the training data).


## Visualizing training proceedure with TensorBoard.
To inspect the training summary one may apply tensorboard.
```terminal
$ ssh -L 6006:127.0.0.1:6006 user@server
```
Then navigate to project folder and start tensorboard.
```terminal
inundation-emulator$ poetry run tensorboard --logdir logs --port 6006
```
Training results for the different runs are visualized at "http://127.0.0.1:6006"

## Notes.
Data folder: T:\Tsunami\PTHA2020_runs_UMA