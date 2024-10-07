import torch

from datasets import load_dataset
from nanosae.core import Data, DataIterator
from typing import Literal
from typing import Iterator, Generator

class HuggingfaceDataIterator(DataIterator):
    def __init__(
        self, 
        data_path: str, 
        *,
        split: str = Literal["train", "validation", "test"],
        streaming: bool = True,
        **kwargs
    ):
        self.data_path = data_path
        self.split = split
        self.streaming = streaming
        self.kwargs = kwargs
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = load_dataset(self.data_path, split=self.split, streaming=self.streaming, **self.kwargs)
        self.iterator = iter(self.dataset)

    def __iter__(self) -> 'HuggingfaceDataIterator':
        return self

    def __next__(self) -> Data:
        try:
            example = next(self.iterator)
            # TODO: un-hardcode this
            return example['input_ids']
        except StopIteration:
            # Reload the dataset and start over
            self._load_dataset()
            return self.__next__()

def truncate(
    tokens_iter: Iterator[torch.Tensor],
    context_size: int,
) -> Generator[torch.Tensor, None, None]:
    """ Truncate an iterator over tokens to a fixed context size """
    for tokens in tokens_iter:
        yield tokens[:context_size]


def batchify(
    tokens_iter: Iterator[torch.Tensor],
    batch_size: int,
) -> Generator[torch.Tensor, None, None]:
    """ Batchify an iterator over tokens
    
    Assumes that each example has the same number of tokens
    """
    batch = []
    for tokens in tokens_iter:
        batch.append(tokens)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch = []
    if len(batch) > 0:
        yield torch.stack(batch, dim = 0)