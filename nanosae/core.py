""" Core types and interfaces """

import torch
import torch.nn as nn 

from typing import Any
from abc import ABC, abstractmethod
from jaxtyping import Float, Int


Tokens = list[str]
ModelActivations = Float[torch.Tensor, "... d_model"]
SAEActivations = Float[torch.Tensor, "... d_sae"]
Loss = Float[torch.Tensor, " ()"]

class SAE(nn.Module, ABC):
    """ Inference-only SAE """
    @abstractmethod
    def encode(self, x: ModelActivations) -> SAEActivations:
        pass

    @abstractmethod
    def decode(self, z: SAEActivations) -> ModelActivations:
        pass

    def forward(self, x: ModelActivations) -> ModelActivations:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

class SAETrainingWrapper(ABC):
    """ Abstract base class for training SAEs """

    sae: SAE

    @abstractmethod
    def training_forward_pass(self, x: ModelActivations) -> tuple[Loss, dict[str, Any]]:
        """ Training forward pass for an SAE

        Inputs: sae, x
        Returns: loss, info_dict
        - loss will generate gradient through `loss.backward()`
        - info_dict contains any additional information needed for logging
        """
        pass

    # Syntactic sugar to allow calling the training forward pass directly
    def __call__(self, x: ModelActivations) -> tuple[Loss, dict[str, Any]]:
        return self.training_forward_pass(x)
    
    def on_train_step_end(self, info_dict: dict[str, Any]) -> None:
        """ Hook for inserting custom logic at the end of a training step 
        
        E.g. post-processing the SAE decoder weights (unit norm, resampling)
        """

class TokensIterator(ABC):
    """ Abstract base class for iterating over a dataset of tokens """ 
    @abstractmethod
    def next_batch(self) -> Tokens:
        pass


class ModelActivationsGetter(ABC):
    """ Abstract base class for getting model activations from tokens """
    @abstractmethod
    def get_activations(self, x: Tokens) -> ModelActivations:
        pass

    # Syntactic sugar to allow calling the model activations getter directly
    def __call__(self, x: Tokens) -> ModelActivations:
        return self.get_activations(x)