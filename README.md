# inundation-emulator
Site specific emulator for tsunami run-up simulations.

# Data folder.
T:\Tsunami\PTHA2020_runs_UMA


# Visulaizing training proceedure with TensorBoard.
To inspect the training summary one may apply tensorboard.
```terminal
$ ssh -L 6006:127.0.0.1:6006 user@server
```
Then navigate to project folder and start tensorboard.
```terminal
inundation-emulator$ poetry run tensorboard --logdir logs --port 6006
```
Training results for the different runs are visualized at "http://127.0.0.1:6006"