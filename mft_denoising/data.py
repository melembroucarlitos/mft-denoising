import torch
import random
import math
from typing import Tuple
from dataclasses import dataclass

set_seed = lambda seed: random.seed(seed)

@dataclass
class DataConfig:
    d: int = 1000
    sparsity: int = 2
    noise_variance: float = 0.03
    n_train: int = 100_000
    n_val: int = 10_000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TwoHotStream:
    """
    Generates B samples with s active coordinates (set to 1) in a d-dimensional vector,
    plus iid Gaussian noise with variance eta2.
    Returns (x, x_star, mask_on) where x_star is the clean two-hot and mask_on is its mask.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        random.seed(torch.initial_seed() & 0xFFFFFFFF)

    @torch.no_grad()
    def sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d, sparsity, eta2 = self.cfg.d, self.cfg.sparsity, self.cfg.noise_variance
        idx = torch.stack([torch.randperm(d, device=self.device)[:sparsity] for _ in range(B)], dim=0)
        x_star = torch.zeros(B, d, device=self.device)
        x_star.scatter_(1, idx, 1.0)
        noise = torch.randn(B, d, device=self.device) * math.sqrt(eta2)
        x = x_star + noise
        return x, x_star

if __name__ == "__main__":
    cfg = DataConfig()
    stream = TwoHotStream(cfg)
    x, x_star = stream.sample_batch(10)
    print(x)
    print(x_star)