import torch

from typing import Any
from tqdm import tqdm
from torch.optim import Optimizer
from dataclasses import dataclass
from nanosae.core import SAETrainingWrapper, TokensIterator, ModelActivationsGetter

@dataclass
class SAETrainerConfig:
    n_train_tokens: int

@dataclass
class TrainStepOutput:
    loss: torch.Tensor
    info_dict: dict[str, Any]
    n_tokens: int

class SAETrainer:
    """ Class for training an SAE """

    sae_train_wrapper: SAETrainingWrapper
    tokens_iterator: TokensIterator
    model_act_getter: ModelActivationsGetter
    optimizer: Optimizer
    config: SAETrainerConfig

    def __init__(
        self,
        config: SAETrainerConfig,
        sae_train_wrapper: SAETrainingWrapper,
        tokens_iterator: TokensIterator,
        model_act_getter: ModelActivationsGetter,
        optimizer: Optimizer,

    ):
        self.config = config
        self.sae_train_wrapper = sae_train_wrapper
        self.tokens_iterator = tokens_iterator
        self.model_act_getter = model_act_getter
        self.optimizer = optimizer

    def train_step(self):
        # Training forward pass
        tokens = self.tokens_iterator.next_batch()
        model_acts = self.model_act_getter(tokens)
        loss, train_info = self.sae_train_wrapper(model_acts)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return TrainStepOutput(loss, train_info, len(tokens))

    def fit(self):
        """ Boilerplate training loop """
        n_tokens_seen = 0
        with tqdm(total = self.config.n_train_tokens) as progress_bar:
            while n_tokens_seen < self.config.n_train_tokens:
                # Perform a training step
                train_step_output = self.train_step()

                # Post-train-step logic
                self.sae_train_wrapper.on_train_step_end(train_step_output.info_dict)

                # TODO: checkpointing logic

                # TODO: evaluation, logging logic

                # Increment progress bar and token count
                n_tokens_seen += train_step_output.n_tokens
                progress_bar.update(train_step_output.n_tokens)