# Soccer Action Spotter

Soccer Action Spotter is a Pytorch-based machine learning application that trains various types of neural network architectures to recognize certain actions in soccer games. The application offers a variety of architectures to choose from, including LSTM, Fully Connected (FC), GRU, Conv1D, and Transformer. 

## Features

- Various types of neural network architectures.
- Integration with Optuna for hyperparameter optimization.
- Integration with Pytorch Lightning for a more structured, easier-to-maintain PyTorch codebase.
- Integration with Weights & Biases for experiment tracking and visualization.

## Installation

You need to have Python 3.6 or later to run Soccer Action Spotter. You can install the required packages using the following command:

```
pip install numpy optuna pytorch_lightning torch wandb
```

## Usage

To train the model, run the script with the `--train` argument:

```
python soccer_action_spotter.py --train
```

To test the model, use the `--test` argument:

```
python soccer_action_spotter.py --test
```

To perform hyperparameter optimization, use the `--hpo` argument:

```
python soccer_action_spotter.py --hpo
```

Additional arguments can be provided to customize the training, testing, and hyperparameter optimization process. For example, you can specify the model type, the dimensions of the input and output, and the learning rate. Here's an example command that trains an LSTM model with an input dimension of 784, a hidden dimension of 256, and an output dimension of 3:

```
python soccer_action_spotter.py --train --model_type LSTM --input_dim 784 --hidden_dim 256 --output_dim 3
```

For a full list of available arguments, you can use the `-h` or `--help` argument:

```
python soccer_action_spotter.py --help
```

## Architecture

Soccer Action Spotter includes several classes of models, which are outlined below:

- `LSTMModel`: A LSTM-based model.
- `FCModel`: A fully connected model.
- `GRUModel`: A GRU-based model.
- `Conv1DModel`: A 1D convolutional model.
- `TransformerModel`: A transformer-based model.

Each of these models can be used with Soccer Action Spotter's PyTorch Lightning `LightningModule`, `SoccerActionSpotter`, for training, validating, and testing.

## Data

The data used for training, validation, and testing should be placed in a directory specified by the `--data_dir` argument. The script assumes that the data is already preprocessed and can be loaded directly into PyTorch tensors.

## Logging

All logs are saved in a directory named `logs/`. These logs can be viewed in Weights & Biases for further analysis and visualization.

## Checkpointing

By default, the script saves checkpoints of the model after each epoch of training. This feature can be disabled with the `--no_checkpoint` argument.

## Hyperparameter Optimization

Soccer Action Spotter uses Optuna for hyperparameter optimization. When the `--hpo` argument is used, the script will perform a number of trials (100 by default) to find the optimal hyperparameters for the model. The results of the best trial will be printed out at the end.

## Contributors

We welcome contributions from the community. If you wish to contribute, please follow the standard Git workflow: fork the repository, make your changes, and submit a pull request.

## License

Soccer Action Spotter is licensed under the MIT license. For more information, see the LICENSE file in the repository.
