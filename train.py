import argparse

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datamodule import SoccerActionDataModule
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging


class LSTMModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)  # output shape is (batch_size, sequence_length, hidden_dim)
        x = self.fc1(output)  # apply the linear layer to each time step
        x = torch.relu(x)
        x = self.fc2(x)
        return x  # x is now of shape (batch_size, sequence_length, output_dim)


class FCModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GRUModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        x = self.fc(output)
        return x


class Conv1DModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, num_heads=4):
        super().__init__()
        self.transformer = nn.Transformer(input_dim, num_heads)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x


class SoccerActionSpotter(pl.LightningModule):
    def __init__(self, model_type, input_dim=784, hidden_dim=256, output_dim=3, learning_rate=0.001):
        super().__init__()
        if model_type == "LSTM":
            model = LSTMModel(input_dim, hidden_dim, output_dim)
        elif model_type == "FC":
            model = FCModel(input_dim, hidden_dim, output_dim)
        elif model_type == "GRU":
            model = GRUModel(input_dim, hidden_dim, output_dim)
        elif model_type == "Conv1D":
            model = Conv1DModel(input_dim, hidden_dim, output_dim)
        elif model_type == "Transformer":
            model = TransformerModel(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def step(self, batch, mode="train"):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        # Calculate accuracy
        y_hat_classes = torch.argmax(y_hat, dim=-1)  # get the predicted classes
        y_classes = y  # true classes
        correct = (y_hat_classes == y_classes).float()  # convert into float for division
        accuracy = correct.sum() / torch.numel(correct)  # divide by the number of elements in y

        # Add the loss and accuracy to the progress bar
        if mode == "train":
            self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{mode}_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log(f"{mode}_loss", loss, on_epoch=True, logger=True)
            self.log(f"{mode}_accuracy", accuracy, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def loss(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))  # flatten the batch and sequence dimensions
        labels = labels.view(-1)  # flatten the batch and sequence dimensions
        return nn.CrossEntropyLoss()(logits, labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Soccer Action Spotter Training Script")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--data_dir", type=str, default="sim_data", help="Path to the dataset directory")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for the sequences")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for the sequences")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--input_dim", type=int, default=784, help="Input dimension for the LSTM model")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for the LSTM model")
    parser.add_argument("--output_dim", type=int, default=3, help="Output dimension for the LSTM model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--swa_epoch_start", type=int, default=1, help="The epoch to start averaging in SWA")
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable model checkpointing")
    parser.add_argument(
        "--model_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "FC", "GRU", "Conv1D", "Transformer"],
        help="Type of the model to use",
    )

    return parser.parse_args()


def train(args):
    # Initialize TensorBoard logger
    wandb_logger = WandbLogger(save_dir="logs/", project="sn-bspotting")

    # Initialize a model checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    # Initialize a learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize a stochastic weight averaging callback
    swa_callback = StochasticWeightAveraging(swa_lrs=args.learning_rate, swa_epoch_start=args.swa_epoch_start)

    # Initialize the data module
    datamodule = SoccerActionDataModule(args.data_dir, args.window_size, args.step_size, args.batch_size)

    # Initialize the model
    model = SoccerActionSpotter(args.model_type, args.input_dim, args.hidden_dim, args.output_dim, args.learning_rate)

    callbacks = (
        [checkpoint_callback, lr_monitor, swa_callback] if not args.no_checkpoint else [lr_monitor, swa_callback]
    )
    # Initialize the trainer
    trainer = pl.Trainer(accelerator="auto", max_epochs=args.epochs, logger=wandb_logger, callbacks=callbacks)

    # Fit the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    trainer.test(datamodule=datamodule)


def test(args):
    raise NotImplementedError("Testing is not implemented yet")


def objective(trial):
    # Suggest values for the hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    window_size = trial.suggest_int("window_size", 1, 10)
    step_size = trial.suggest_int("step_size", 1, 5)

    # Initialize the model
    model = SoccerActionSpotter(args.model_type, args.input_dim, hidden_dim, args.output_dim, learning_rate)

    # Initialize callbacks
    wandb_logger = WandbLogger("logs/", project="sn-bspotting")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    swa_callback = StochasticWeightAveraging(swa_lrs=learning_rate, swa_epoch_start=args.swa_epoch_start)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")  # Add a pruning callback
    callbacks = [checkpoint_callback, lr_monitor, swa_callback, pruning_callback]

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    # Initialize the data module
    datamodule = SoccerActionDataModule(args.data_dir, window_size, step_size, args.batch_size)

    # Fit the model
    trainer.fit(model, datamodule=datamodule)

    # Return the best validation loss
    return trainer.callback_metrics["val_loss"].item()


def hpo(args):
    # Define a study and optimize the objective function
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage=storage_name
    )
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial_ = study.best_trial
    print(f"  Value: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.hpo:
        hpo(args)
    else:
        raise ValueError("You must either specify --train or --hpo")
