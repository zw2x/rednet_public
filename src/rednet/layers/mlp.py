import torch
import torch.nn as nn
from einops import repeat
from ..common_utils import mask_mean


class SwiGLU(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.act_fn = nn.SiLU()
        self.gate_proj = nn.Linear(dim, dim, bias=bias)
        self.lin = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        return self.lin(x) * self.act_fn(self.gate_proj(x))


def get_act_fn(act_fn):
    if act_fn == "silu":
        act_fn = nn.SiLU()
    elif act_fn == "gelu":
        act_fn = nn.GELU()
    elif act_fn == "leaky_relu":
        act_fn = nn.LeakyReLU(negative_slope=0.1)
    elif act_fn == "relu":
        act_fn = nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    return act_fn


class Mlp(nn.Module):
    # llama/af3 mlp
    def __init__(
        self,
        dim,
        out_dim=None,
        expansion_factor=4,
        apply_prenorm=True,
        depth=1,
        embed_dim=None,
        zero_init=False,
        act_fn: str = "silu",
    ):
        super().__init__()
        self.apply_prenorm = apply_prenorm
        self.norm = nn.LayerNorm(dim) if apply_prenorm else nn.Identity()
        out_dim = out_dim or dim
        embed_dim = embed_dim or dim
        hid_dim = embed_dim * expansion_factor
        self.up_proj = nn.Linear(dim, hid_dim, bias=False)
        self.down_proj = nn.Linear(hid_dim, out_dim, bias=False)

        self.act_fn_str = act_fn
        if act_fn == "silu":
            self.gate_proj = nn.Linear(dim, hid_dim, bias=False)
        self.act_fn = get_act_fn(act_fn)

        if depth > 1:
            self.hid_modules = nn.ModuleList(
                [nn.ModuleList([nn.Linear(hid_dim, hid_dim, bias=False), get_act_fn(act_fn)]) for _ in range(depth - 1)]
            )
            self.hid_depth = depth - 1
        else:
            self.hid_depth = 0

        self.init_weights(zero_init)

    def init_weights(self, zero_init=False):
        if zero_init:
            nn.init.zeros_(self.down_proj.weight)
        else:
            nn.init.trunc_normal_(self.down_proj.weight, std=0.02)

        if self.act_fn_str == "silu":
            nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)

        nn.init.trunc_normal_(self.up_proj.weight, std=0.02)

        if self.apply_prenorm:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

        if self.hid_depth > 0:
            for lin, act in self.hid_modules:
                nn.init.trunc_normal_(lin.weight, std=0.02)

    def forward(self, x):
        x = self.norm(x)
        if self.act_fn_str == "silu":
            h = self.act_fn(self.gate_proj(x)).mul(self.up_proj(x))
        else:
            h = self.act_fn(self.up_proj(x))
        if self.hid_depth > 0:
            for lin, act in self.hid_modules:
                h = act(lin(h))
        x = self.down_proj(h)
        return x
