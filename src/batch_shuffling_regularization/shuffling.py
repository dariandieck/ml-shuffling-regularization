import torch

# A batch order function returns the training indices for one epoch,
# shaped (n_batches, batch_size): row i holds the samples of the i-th step.


def no_shuffle(n: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    return torch.arange(n).reshape(-1, batch_size)
