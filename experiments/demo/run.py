# %%

import torch

from datasets import load_dataset
from typing import Literal

from transformer_lens import HookedTransformer
from nanosae.core import Tokens, TokensIterator
from nanosae.core import ModelActivations, ModelActivationsGetter
from nanosae.train import SAETrainerConfig, TrainStepOutput, SAETrainer

from nanosae.sae.vanilla import VanillaSAE, VanillaSAETrainingWrapper

# %%
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
    

data_iter = TinystoriesTokensIterator("roneneldan/TinyStories", batch_size = 32)
batch = data_iter.next_batch()
print(len(batch))
print(batch[0])

# %% 
class TransformerLensActivationsGetter(ModelActivationsGetter):
    def __init__(self, model_path: str, hook_name: str):
        self.model = HookedTransformer.from_pretrained(model_path)
        self.hook_name = hook_name
    
    def get_activations(self, tokens: Tokens) -> ModelActivations:
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        return cache[self.hook_name]
    
model_act_getter = TransformerLensActivationsGetter("solu-1l", "blocks.0.hook_resid_pre")
model_acts = model_act_getter(batch)
print(model_acts.shape)

# %%

sae = VanillaSAE(d_in = 512, d_sae = 1024)
sae_train_wrapper = VanillaSAETrainingWrapper(sae, l1_coeff = 0.2)

# %%

training_config = SAETrainerConfig(n_train_tokens = 100_000)
trainer = SAETrainer(
    config = training_config,
    sae_train_wrapper = sae_train_wrapper,
    tokens_iterator = data_iter,
    model_act_getter = model_act_getter,
    optimizer = torch.optim.Adam(sae.parameters(), lr = 1e-3)
)

# %%
trainer.fit()