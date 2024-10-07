""" Script to reproduce https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb 

Usage: `python run.py`

For detailed usage: `python run.py --help`
"""

import torch

from dataclasses import dataclass
from typing import Iterator

from transformer_lens import HookedTransformer
from simple_parsing import ArgumentParser

from nanosae.core import Data, Tokens
from nanosae.model.tlens import TransformerLensActivationsGetter
from nanosae.train import SAETrainerConfig, SAETrainer
from nanosae.sae.vanilla import VanillaSAE, VanillaSAETrainingWrapper
from nanosae.logging import WandbLogger, WandbConfig
from nanosae.data import HuggingfaceDataIterator, batchify, truncate


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

def setup_trainer(config: ExperimentConfig) -> SAETrainer:
    # Setup model
    model_act_getter = TransformerLensActivationsGetter(
        model_path=config.model_path, hook_name=config.hook_name
    )

    # Setup data
    data_iterator = HuggingfaceDataIterator(
        data_path=config.data_path, split=config.split, streaming=config.streaming
    )

    def iter_tokens(data_iter: Iterator[Data]) -> Iterator[Tokens]:
        for data in data_iter:
            yield model_act_getter.get_tokens(data)

    tokens_iterator = iter_tokens(data_iterator)
    tokens_iterator = truncate(tokens_iterator, context_size=config.context_size)
    tokens_iterator = batchify(tokens_iterator, batch_size=config.batch_size)

    # Setup SAE
    sae = VanillaSAE(
        d_in=model_act_getter.d_model,
        d_sae=model_act_getter.d_model * config.expansion_factor,
    )
    sae_train_wrapper = VanillaSAETrainingWrapper(sae, l1_coeff=config.l1_coeff)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=config.lr, betas=(config.adam_beta1, config.adam_beta2)
    )

    # Setup logging
    wandb_config = WandbConfig(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_group,
        name=config.wandb_name,
        mode=config.wandb_mode,
    )
    logger = WandbLogger(wandb_config)
    logger.set_config(config)

    # Setup training
    training_config = SAETrainerConfig(
        n_train_tokens=config.n_train_tokens,
        checkpoint_every_n_tokens=config.checkpoint_every_n_tokens,
    )

    return SAETrainer(
        config=training_config,
        sae_train_wrapper=sae_train_wrapper,
        tokens_iterator=tokens_iterator,
        model_act_getter=model_act_getter,
        optimizer=optimizer,
        logger=logger,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    config = args.config
    trainer = setup_trainer(config)
    trainer.fit()
