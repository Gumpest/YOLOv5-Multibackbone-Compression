import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)

        self.attention_norm = nn.LayerNorm(c, eps=1e-6)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

        self.ffn_norm = nn.LayerNorm(c, eps=1e-6)
        self.ffn = Mlp(c)

    def forward(self, x):
        h = x
        x = self.attention_norm(x) # add
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + h

        h = x
        x = self.ffn_norm(x)       # add
        x = self.ffn(x) + h
        return x

class Mlp(nn.Module):
    def __init__(self, c):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(c, 5120)
        self.fc2 = nn.Linear(5120, c)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Encoder(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        # print(x.shape)
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

x = torch.ones(1, 1280, 8, 8)
m = Encoder(1280, 1280, 16, 3)
r = m(x)