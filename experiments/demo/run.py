# %%

import torch

from dataclasses import dataclass
from datasets import load_dataset
from typing import Literal

from transformer_lens import HookedTransformer
from nanosae.core import Tokens, TokensIterator
from nanosae.core import ModelActivations, ModelActivationsGetter
from nanosae.train import SAETrainerConfig, TrainStepOutput, SAETrainer
from nanosae.sae.vanilla import VanillaSAE, VanillaSAETrainingWrapper
from nanosae.logging import WandbLogger, WandbConfig

dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2"
dataset = load_dataset(dataset_path, split="train", streaming=True)
for example in dataset:
    print(example['input_ids'])
    break

class TinystoriesTokensIterator(TokensIterator):
    def __init__(self, data_path: str, *, batch_size: int = 1, split: Literal["train", "test"] = "train"):
        self.dataset = load_dataset(data_path, split=split)
        self.idx = 0
        self.batch_size = batch_size

    def next_batch(self) -> Tokens:
        batch = self.dataset[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        if self.idx >= len(self.dataset):
            self.idx = 0
        return batch["text"]

class TransformerLensActivationsGetter(ModelActivationsGetter):
    def __init__(self, model_path: str, hook_name: str):
        self.model = HookedTransformer.from_pretrained(model_path)
        self.hook_name = hook_name
    
    def get_activations(self, tokens: Tokens) -> ModelActivations:
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        return cache[self.hook_name]

# %%

@dataclass 
class ExperimentConfig:
    # Data and model details
    data_path: str = "apollo-research/roneneldan-TinyStories-tokenizer-gpt2"
    model_path: str = "tiny-stories-1L-21M"
    hook_name: str = "blocks.0.hook_mlp_out"

    # Training details 
    batch_size = 4096
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    n_train_tokens: int = 100_000_000
    l1_coeff: float = 5
    lr: float = 5e-5

    # Boilerplate logging details
    checkpoint_every_n_tokens: int
    wandb_project: str
    wandb_entity: str
    wandb_group: str
    wandb_name: str
    wandb_mode: str = "online"

def run_experiment():

    data_iter = TinystoriesTokensIterator("roneneldan/TinyStories", batch_size = 4)
    batch = data_iter.next_batch()
    print(len(batch))
    print(batch[0])

    model_act_getter = TransformerLensActivationsGetter("tiny-stories-1L-21M", "blocks.0.hook_resid_pre")
    model_acts = model_act_getter(batch)
    print(model_acts.shape)

    sae = VanillaSAE(d_in = 512, d_sae = 1024)
    sae_train_wrapper = VanillaSAETrainingWrapper(sae, l1_coeff = 0.2)


    wandb_config = WandbConfig(project = "nanosae", entity = "dtch1997", group = "demo", name = "demo")

    logger = WandbLogger(wandb_config)
    training_config = SAETrainerConfig(n_train_tokens = 10_000_000, checkpoint_every_n_tokens = 10_000)
    trainer = SAETrainer(
        config = training_config,
        sae_train_wrapper = sae_train_wrapper,
        tokens_iterator = data_iter,
        model_act_getter = model_act_getter,
        optimizer = torch.optim.Adam(sae.parameters(), lr = 1e-3),
        logger = logger
    )
    trainer.fit()

if __name__ == "__main__":
    run_experiment()