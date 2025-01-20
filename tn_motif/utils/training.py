"""
Code for training PyTorch models with k-fold cross validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import numpy as np


class ModelTraining:
    def __init__(
        self,
        model_class: nn.Module,
        model_params: dict,
        dataset: Dataset,
        # TODO: combine the following into training params dict
        criterion: str,
        optimizer: str,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=12,
        k_folds=5,
        device=None,
        verbose=True,
    ):
        """
        Initializes the ModelTraining class.

        Args:
            model_class (class): PyTorch model class.
            model_params (dict): parameters for defining model
            dataset (torch.utils.data.Dataset): The full dataset.
            criterion (str): Criterion to use for calculating loss
            optimizer (str): Optimizer name, instantiated when model is called
            batch_size (int): Batch size for DataLoader.
            learning_rate (float): Learning rate for optimizer.
            num_epochs (int): Number of epochs for training.
            k_folds (int): Number of folds for cross-validation.
            device (torch.device or None): Device to train on. Defaults to CUDA if available.
            verbose (bool): If True, prints training progress.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose
        # Note that we will only instantiate the optimizer when it comes to model training
        self.optimizer = optimizer
        self.criterion = self._criterion_function(criterion)

    def _criterion_function(self, criterion: str):
        """
        Return nn criterion function given name
        """
        if criterion.lower() == "mae":
            return nn.L1Loss()
        elif criterion.lower() == "mse":
            return nn.MSELoss()
        else:
            raise NotImplementedError(
                f"{criterion} not implemented; valid options are mae and mse"
            )

    def _instantiate_optimizer(self, model):
        """
        Instantiate optimizer name when creating model object
        """
        if self.optimizer == "adam":
            return torch.optim.Adam(params=model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(params=model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Optimizers can only be ada")

    def train_final_model(self):
        """
        Trains the final model on the entire training set.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            model (nn.Module): The model to train.
            criterion: Loss function.
            optimizer: Optimizer.
        """
        # setting things up
        model = self.model_class(self.model_params)
        criterion = self.criterion
        optimizer = self._instantiate_optimizer(model=model)
        model.to(self.device)
        # the dataset loader
        train_loader_full = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.num_epochs):
            epoch_loss = self.training_loop(
                train_loader_full, model, criterion, optimizer
            )
            if self.verbose:
                print(
                    f"Final Model - Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}"
                )

        # set the trained model attribute
        self.trained_model = model
        return model

    def _training_loop(self, train_loader, model, criterion, optimizer):
        """
        Executes the training loop over the dataset for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            model (nn.Module): The model to train.
            criterion: Loss function.
            optimizer: Optimizer.

        Returns:
            float: Average training loss for the epoch.
        """
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def _evaluate_model(self, data_loader, model, criterion):
        """
        Evaluates the model on a given dataset.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            model (nn.Module): The trained model.
            criterion: Loss function.

        Returns:
            dict: Pearson_R, Spearman_R, Loss, as keys
        """
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().unsqueeze(1)

                outputs = model(inputs).squeeze(
                    1
                )  # Assuming model outputs shape [batch, 1]

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.squeeze(1).cpu().numpy())
                loss = criterion(outputs, targets.squeeze(1))
                # keep tracking loss over the epochs
                total_loss += loss.item() * inputs.size(0)

        pearson_r, _ = pearsonr(all_targets, all_preds)
        spearman_r, _ = spearmanr(all_targets, all_preds)
        average_loss = total_loss / len(data_loader.dataset)

        metrics_dict = {
            "pearson_r": pearson_r,
            "spearman_rho": spearman_r,
            "loss": average_loss,
        }

        return metrics_dict

    def train_val_model(self, train_loader, val_loader, model, criterion, optimizer):
        """
        Runs the training process and evaluates the model on validation data.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            model (nn.Module): The model to train.
            criterion: Loss function.
            optimizer: Optimizer.

        Returns:
            train_losses (list(float)): losses by epoch of training
            val_metrics (list(dict(float))): metrics by epoch of training
        """
        train_losses = []
        val_metrics = []

        for epoch in range(self.num_epochs):
            epoch_loss = self._training_loop(train_loader, model, criterion, optimizer)
            train_losses.append(epoch_loss)

            if self.verbose:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}"
                )

            # evaluate model on validation set
            val_metrics_epoch = self._evaluate_model(val_loader, model, criterion)

            if self.verbose:
                val_loss, pearson_r, spearman_r = (
                    val_metrics_epoch["loss"],
                    val_metrics_epoch["pearson_r"],
                    val_metrics_epoch["spearman_rho"],
                )
                print(
                    f"    Validation Loss: {val_loss:.4f}, Pearson R: {pearson_r:.4f}, Spearman R: {spearman_r:.4f}"
                )
            val_metrics.append(val_metrics_epoch)

        return train_losses, val_metrics

    def cross_validation(self):
        """
        Performs k-fold cross-validation on the training set.

        """
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        val_metrics = {}
        train_losses = {}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset), start=1):
            if self.verbose:
                print(f"\n--- Fold {fold}/{self.k_folds} ---")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False
            )

            # instantiate model and optimizer for this fold of training
            model = self.model_class(self.model_params)
            model.to(self.device)
            criterion = self.criterion
            optimizer = self._instantiate_optimizer(model=model)

            # use the train_model method to get train losses and validation
            # metrics for this fold
            train_losses_fold, val_metrics_fold = self.train_val_model(
                train_loader, val_loader, model, criterion, optimizer
            )

            # save in a dictionary
            train_losses[f"fold {fold}"] = train_losses_fold
            val_metrics[f"fold {fold}"] = val_metrics_fold

        # set attributes for access at later timepoints
        self.train_losses = train_losses
        self.val_metrics = val_metrics

        # keeping this here in case I decide to return the metrics as well
        return train_losses, val_metrics

    def evaluate_test(self, test_loader, model):
        """
        Evaluates the model on the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            model (nn.Module): The trained model.

        Returns:
            tuple: (Pearson R, Spearman R)
        """
        metrics_dict = self._evaluate_model(test_loader, model)
        if self.verbose:
            print(f"Test Loss: {metrics_dict['loss']:.4f}")
            print(f"Test Pearson r: {metrics_dict['pearson_r']:.4f}")
            print(f"Test Spearman rho: {metrics_dict['spearman_r']:.4f}")
        return metrics_dict
