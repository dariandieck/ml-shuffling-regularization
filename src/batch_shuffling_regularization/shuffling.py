import torch

# A batch order function returns the training indices for one epoch,
# shaped (n_batches, batch_size): row i holds the samples of the i-th step.


def no_shuffle(n: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    return torch.arange(n).reshape(-1, batch_size)


def shuffle_batch_order(n: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    batches = no_shuffle(n, batch_size, gen)
    return batches[torch.randperm(batches.shape[0], generator=gen)]


def shuffle_within_batches(n: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    batches = no_shuffle(n, batch_size, gen)
    # argsort of uniform noise gives an independent random permutation for every row
    row_perms = torch.rand(batches.shape, generator=gen).argsort(dim=1)
    return batches.gather(1, row_perms)


def full_reshuffle(n: int, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=gen).reshape(-1, batch_size)


VARIANTS = {
    "A": no_shuffle,
    "B": shuffle_batch_order,
    "C": shuffle_within_batches,
    "D": full_reshuffle,
}
