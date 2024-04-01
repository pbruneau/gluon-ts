# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.model.lightning_util import has_validation_loop

from .module import DeepARModel
import pdb

def jeffreys_divergence(mu_P, variance_P, mu_Q, variance_Q):
    part1 = ((variance_P + (mu_P - mu_Q)**2) / (2 * variance_Q))
    part2 = ((variance_Q + (mu_Q - mu_P)**2) / (2 * variance_P))
    jeffreys = part1 + part2 - 1
    return jeffreys

def compute_and_log_jeffreys_divergences(distr, logger, global_step):
    n_components = len(distr.components)
    all_divergences = []

    # Compute Jeffreys divergence for each pair of components
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Ensure a != b and exploit symmetry
            mu_P = distr.components[i].base_dist.mean
            sigma_P = torch.sqrt(distr.components[i].base_dist.variance)
            mu_Q = distr.components[j].base_dist.mean
            sigma_Q = torch.sqrt(distr.components[j].base_dist.variance)

            # Compute Jeffreys divergence for all positions
            jeffreys = jeffreys_divergence(mu_P, sigma_P, mu_Q, sigma_Q)
            
            # Flatten and store the computed divergences
            all_divergences.append(jeffreys.flatten())

    # Concatenate all divergences into a single tensor for histogram logging
    all_divergences = torch.cat(all_divergences)

    # Log the distribution of Jeffreys divergences as a histogram
    if logger and hasattr(logger, 'experiment') and all_divergences.numel() > 0:
        logger.experiment.add_histogram("Jeffreys Divergence", all_divergences, global_step=global_step)


class DeepARLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``DeepARModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``DeepARModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``DeepARModel`` to be trained.
    loss
        Loss function to be used for training.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    patience
        Patience parameter for learning rate scheduler.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepARModel(**model_kwargs)
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss, distr = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
            loss=self.loss,
        )

        #pdb.set_trace()
        #if self.logger and isinstance(self.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter):
        #    self.logger.experiment.add_histogram("Train/Loss", train_loss, global_step=self.global_step)
        compute_and_log_jeffreys_divergences(distr, self.logger, self.global_step)
        
        # distr.components holds: [AffineTransformed(), AffineTransformed(), AffineTransformed()]
        # each AffineTransformed holds: base_dist.mean and base_dist.variance, then transformed with loc and scale
        # each variable is a [batch_size, nsteps] Tensor
        
        train_loss = train_loss.mean()
        
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
            loss=self.loss,
        ).mean()

        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = (
            "val_loss" if has_validation_loop(self.trainer) else "train_loss"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }
