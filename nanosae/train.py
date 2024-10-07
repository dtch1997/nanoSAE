from __future__ import annotations

import torch

from typing import Any
from tqdm import tqdm
from torch.optim import Optimizer
from dataclasses import dataclass
from nanosae.core import SAETrainingWrapper, TokensIterator, ModelActivationsGetter, TrainStepOutput
from nanosae.logging import Logger

@dataclass
class SAETrainerConfig:
    n_train_tokens: int
    checkpoint_every_n_tokens: int
    disable_tqdm: bool = False

@dataclass
class SAETrainer:
    """ Class for training an SAE """

    sae_train_wrapper: SAETrainingWrapper
    tokens_iterator: TokensIterator
    model_act_getter: ModelActivationsGetter
    optimizer: Optimizer
    config: SAETrainerConfig
    logger: Logger

    def train_step(self):
        # Training forward pass
        tokens = self.tokens_iterator.next_batch()
        model_acts = self.model_act_getter(tokens)
        train_step_output = self.sae_train_wrapper(model_acts)

        # Backward pass
        loss = train_step_output.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_step_output

    def fit(self):
        """ Boilerplate training loop """
        n_tokens_seen = 0

        with tqdm(total = self.config.n_train_tokens) as progress_bar:
            while n_tokens_seen < self.config.n_train_tokens:
                # Perform a training step
                train_step_output = self.train_step()

                # Post-train-step logic
                self.sae_train_wrapper.on_train_step_end()

                # TODO: checkpointing logic

                log_dict = self._build_train_step_log_dict(train_step_output, n_tokens_seen)
                self.logger.log(log_dict, step  = n_tokens_seen)

                # Increment progress bar and token count
                n_tokens_seen += train_step_output.n_tokens
                progress_bar.update(train_step_output.n_tokens)

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
    ) -> dict[str, Any]:
        """" Build a dictionary of things to log after a training step """

        # Unpack train step output
        sae_in = output.sae_in
        sae_out = output.sae_out
        sae_act = output.sae_act
        loss = output.loss.item()

        # Get the current learning rate
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute additional SAE metrics not covered in loss
        l0 = (sae_act > 0).float().sum(-1).mean()
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        # Build log dict
        log_dict = {}

        # Add custom losses
        log_dict["losses/total_loss"] = loss
        for k, v in output.loss_dict.items():
            log_dict[f"losses/{k}"] = v.mean().item()

        # Add metrics
        log_dict.update({            
            # variance explained
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
            # sparsity
            # "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            # "sparsity/dead_features": self.dead_neurons.sum().item(),
            # other training details
            "details/current_learning_rate": current_learning_rate,
            # "details/current_l1_coefficient": self.current_l1_coefficient,
            "details/n_training_tokens": n_training_tokens,
        })

        return log_dict