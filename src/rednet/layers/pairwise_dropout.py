import torch
import torch.nn as nn


class PairwiseDropout(nn.Module):
    def __init__(self, p=0.0, orientation="row"):
        super().__init__()
        self.p = p
        self.orientation = orientation

    def forward(self, x: torch.Tensor):
        # x: (bsz, nrows, ncols, ...)
        if not self.training or self.p == 0:
            return x
        shape = list(x.shape)
        if self.orientation == "row":  # every row of the same batch shares the same mask
            shape[1] = 1
        else:
            shape[2] = 1
        mask = torch.rand(shape, device=x.device) <= self.p
        return x.masked_fill(mask, 0.0)
