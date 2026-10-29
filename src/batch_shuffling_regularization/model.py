import torch
from torch import nn


def build_mlp(seed: int, device: torch.device, hidden: tuple[int, ...] = (256, 256)) -> nn.Sequential:
    # The seed fixes the weight initialisation, so every variant can start from identical weights.
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    n_in = 28 * 28
    for n_out in hidden:
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    layers.append(nn.Linear(n_in, 10))
    return nn.Sequential(*layers).to(device)
