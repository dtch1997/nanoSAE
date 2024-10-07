# %%

""" Reproduction of https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb """

import torch

from dataclasses import dataclass
from typing import Iterator

from transformer_lens import HookedTransformer
from nanosae.core import Data, Tokens
from nanosae.core import ModelActivations, ModelActivationsGetter
from nanosae.train import SAETrainerConfig, SAETrainer
from nanosae.sae.vanilla import VanillaSAE, VanillaSAETrainingWrapper
from nanosae.logging import WandbLogger, WandbConfig
from nanosae.utils.device import get_device
from nanosae.data import HuggingfaceDataIterator, batchify, truncate

class TransformerLensActivationsGetter(ModelActivationsGetter):

    model: HookedTransformer
    hook_name: str
    device: str

    def __init__(self, model_path: str, hook_name: str, device = None):
        if device is None:
            device = get_device()

        self.model = HookedTransformer.from_pretrained(model_path)
        self.hook_name = hook_name
        self.device = device

    @property
    def d_model(self):
        return self.model.cfg.d_model
    
    def get_tokens(self, data: Data) -> Tokens:
        if isinstance(data, str):
            return self.model.to_tokens(data)
        elif isinstance(data, list) and isinstance(data[0], int):
            return torch.tensor(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def get_activations(self, tokens: Tokens) -> ModelActivations:
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        return cache[self.hook_name]

@dataclass 
class ExperimentConfig:
    # Model details
    model_path: str = "tiny-stories-1L-21M"
    hook_name: str = "blocks.0.hook_mlp_out"

    # Dataset details
    data_path: str = "apollo-research/roneneldan-TinyStories-tokenizer-gpt2"
    split: str = "train"
    streaming: bool = True

    # SAE architecture details
    expansion_factor: int = 16

    # Training details 
    batch_size = 8
    context_size = 512
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    n_train_tokens: int = 100_000_000
    l1_coeff: float = 5
    lr: float = 5e-5

    # Boilerplate logging details
    checkpoint_every_n_tokens: int = 1_000_000
    wandb_project: str = "nanosae"
    wandb_entity: str = "dtch1997"
    wandb_group: str = "demo"
    wandb_name: str = "demo"
    wandb_mode: str = "online"

config = ExperimentConfig()

# %%

def setup_trainer(config: ExperimentConfig) -> SAETrainer:

    # Setup model 
    model_act_getter = TransformerLensActivationsGetter(
        model_path = config.model_path,
        hook_name = config.hook_name
    )

    # Setup data
    data_iterator = HuggingfaceDataIterator(
        data_path = config.data_path,
        split = config.split,
        streaming = config.streaming
    )

    def iter_tokens(data_iter: Iterator[Data]) -> Iterator[Tokens]:
        for data in data_iter:
            yield model_act_getter.get_tokens(data)

    tokens_iterator = iter_tokens(data_iterator)
    tokens_iterator = truncate(tokens_iterator, context_size=config.context_size)
    tokens_iterator = batchify(tokens_iterator, batch_size=config.batch_size)

    # Setup SAE
    sae = VanillaSAE(d_in = model_act_getter.d_model, d_sae = model_act_getter.d_model * config.expansion_factor)
    sae_train_wrapper = VanillaSAETrainingWrapper(sae, l1_coeff = config.l1_coeff)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        sae.parameters(), 
        lr = config.lr, 
        betas = (config.adam_beta1, config.adam_beta2)
    )

    # Setup logging
    wandb_config = WandbConfig(
        project = config.wandb_project,
        entity = config.wandb_entity,
        group = config.wandb_group,
        name = config.wandb_name,
        mode = config.wandb_mode
    )
    logger = WandbLogger(wandb_config)
    logger.set_config(config)

    # Setup training
    training_config = SAETrainerConfig(
        n_train_tokens = config.n_train_tokens,
        checkpoint_every_n_tokens = config.checkpoint_every_n_tokens
    )

    return SAETrainer(
        config = training_config,
        sae_train_wrapper = sae_train_wrapper,
        tokens_iterator = tokens_iterator,
        model_act_getter = model_act_getter,
        optimizer = optimizer,
        logger = logger
    )
    

if __name__ == "__main__":
    trainer = setup_trainer(config)
    trainer.fit()