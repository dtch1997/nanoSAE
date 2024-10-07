import torch

from transformer_lens import HookedTransformer
from nanosae.core import ModelActivations, ModelActivationsGetter, Data, Tokens
from nanosae.utils.device import get_device


class TransformerLensActivationsGetter(ModelActivationsGetter):
    model: HookedTransformer
    hook_name: str
    device: str

    def __init__(self, model_path: str, hook_name: str, device=None):
        if device is None:
            device = get_device()

        self.model = HookedTransformer.from_pretrained(model_path).to(device)
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
