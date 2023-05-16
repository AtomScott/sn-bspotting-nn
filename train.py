import argparse
import math
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamodule import SoccerActionDataModule
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        cosine = F.linear(F.normalize(logits), F.normalize(labels))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        labels = labels.view(-1, 1)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=3, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)  # output shape is (batch_size, sequence_length, hidden_dim * 2)
        output = self.dropout(output)
        x = self.fc1(output)  # apply the linear layer to each time step
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x  # x is now of shape (batch_size, sequence_length, output_dim)

# class FCModel(nn.Module):
#     def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, dropout=0.5):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
#         self.fc4 = nn.Linear(hidden_dim // 4, output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.LeakyReLU()

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = self.activation(self.fc1(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc2(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc3(x))
#         x = self.dropout(x)
#         x = self.fc4(x)
#         x = torch.sigmoid(x)
#         return x


# class GRUModel(nn.Module):
#     def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, dropout=0.5):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

#     def forward(self, x):
#         output, _ = self.gru(x)  # output shape is (batch_size, sequence_length, hidden_dim * 2)
#         output = self.dropout(output)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
#         return x  # x is now of shape (batch_size, sequence_length, output_dim)

class Conv1DModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is B x L x C whereas conv1d expects B x C x L
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        
        # Convert back to B x L x C
        x = x.transpose(1, 2)
        
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=3, num_heads=4, dropout=0.5, num_layers=1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
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
        self.loss_function = FocalLoss()

    def step(self, batch, mode="train"):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        # Calculate accuracy
        y_hat_classes = torch.argmax(y_hat, dim=-1)  # get the predicted classes
        y_classes = torch.argmax(y, dim=-1)  # true classes
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
        return self.loss_function(logits, labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Soccer Action Spotter Training Script")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the dataset directory")
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
    name = f"{args.model_type}_{args.window_size}_{args.step_size}_{args.batch_size}_{args.learning_rate}"
    wandb_logger = WandbLogger(name=name, save_dir="logs/", project="sn-bspotting")

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
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    window_size = trial.suggest_int("window_size", 1, 256, log=True)
    step_size = trial.suggest_int("step_size", 1, window_size, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 1, 512, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    model_type = trial.suggest_categorical("model_type", ["LSTM", "Conv1D", "Transformer"])
    
    # Initialize the model
    model = SoccerActionSpotter(model_type, args.input_dim, hidden_dim, args.output_dim, learning_rate)

    # Initialize callbacks
    name = f"{model_type}_{window_size}_{step_size}_{batch_size}_{learning_rate:.1g}_{hidden_dim}"
    wandb_logger = WandbLogger(name=name, save_dir="logs/", project="sn-bspotting")

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
    datamodule = SoccerActionDataModule(args.data_dir, window_size, step_size, batch_size)

    # Fit the model
    trainer.fit(model, datamodule=datamodule)

    # Return the best validation loss
    return trainer.callback_metrics["val_loss"].item()


def hpo(args):
    # Define a study and optimize the objective function
    study_name = "hpo_lstm_conv1d_transformer"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage=storage_name, load_if_exists=True
    )
    study.optimize(objective, n_trials=1)

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
