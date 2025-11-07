import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_b_c_t: torch.Tensor) -> torch.Tensor:
        B, C, T = x_b_c_t.shape
        x = x_b_c_t.permute(0, 2, 1).contiguous()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            return t.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, C)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        out = (out + x).permute(0, 2, 1).contiguous()
        return out


class BaseDTA(nn.Module):
    """
    Pure DTA Block that maps X -> F_out with the same channel count.
    Pre-Conv and Post-Conv with residual concat are handled outside this block.
    Assumptions:
      - The feature map is split into 4 windows: h = H/2, w = W/2 (n=4).
      - linear_h_dim = h = w; linear_H_dim = H at this scale.
    """
    def __init__(self, inplanes: int,
                 linear_h_dim: int, linear_H_dim: int,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        # Token projectors for Local Tokens Up-sample
        self.linear_row = nn.Linear(1, linear_h_dim, bias=True)
        self.linear_col = nn.Linear(1, linear_h_dim, bias=True)

        # Global Token expansion and fusion
        self.linear_GT = nn.Linear(1, linear_H_dim, bias=True)
        self.conv_fuse_GT = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.mhsa = MHSA(embed_dim=inplanes, num_heads=num_heads, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.linear_h_dim = linear_h_dim
        self.linear_H_dim = linear_H_dim

    @staticmethod
    def _pad_to_even(x: torch.Tensor):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad_hw):
        pad_h, pad_w = pad_hw
        if pad_h == 0 and pad_w == 0:
            return x
        _, _, H, W = x.shape
        return x[:, :, :H - pad_h, :W - pad_w]

    def _local_mhsa(self, y_in_b_c_t: torch.Tensor) -> torch.Tensor:
        return self.mhsa(y_in_b_c_t)

    def _local_token_upsample(self, lt_b_c_1_1: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, C, _, _ = lt_b_c_1_1.shape
        assert h == self.linear_h_dim and w == self.linear_h_dim, "window size must match linear_h_dim"
        row = self.linear_row(lt_b_c_1_1).view(B, C, h, 1)
        col = self.linear_col(lt_b_c_1_1).view(B, C, 1, w)
        w_lt = self.sigmoid(row * col)
        return w_lt

    def _global_token_attention(self, f_wlt: torch.Tensor, gt_tokens_list, H: int, W: int) -> torch.Tensor:
        B, C, _, _ = f_wlt.shape
        assert H == self.linear_H_dim and W == self.linear_H_dim, "window size must match linear_H_dim"

        gt_expanded = []
        for gt_i in gt_tokens_list:
            gt_1_1 = gt_i.view(B, C, 1, 1)
            gt_h = self.linear_GT(gt_1_1)
            gt_h = gt_h.permute(0, 1, 3, 2).contiguous()
            gt_expanded.append(gt_h)

        w_gt_cat = torch.cat(gt_expanded, dim=3)
        w_gt = self.conv_fuse_GT(w_gt_cat)

        q = f_wlt.view(B, C, -1).permute(0, 2, 1).contiguous()
        k = w_gt.view(B, C, -1).permute(0, 2, 1).contiguous()
        v = k

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_hw = self._pad_to_even(x)

        B, C, H, W = x.shape
        h, w = H // 2, W // 2
        assert h == self.linear_h_dim and w == self.linear_h_dim, "window size must equal linear_h_dim"
        assert H == self.linear_H_dim and W == self.linear_H_dim, "window size must match linear_H_dim"

        gt = self.avg_pool(x).view(B, C, -1)

        x_top, x_bottom = torch.chunk(x, 2, dim=2)
        x11, x12 = torch.chunk(x_top, 2, dim=3)
        x21, x22 = torch.chunk(x_bottom, 2, dim=3)
        quads = [x11, x12, x21, x22]

        x_out_quads = []
        gt_tokens = []

        for q_map in quads:
            lt_i = self.avg_pool(q_map).view(B, C, -1)
            x_hw = q_map.reshape(B, C, -1)
            y_in = torch.cat([x_hw, lt_i, gt], dim=2)
            y_out = self._local_mhsa(y_in)

            x_prime, lt_prime, gt_prime = torch.split(y_out, [h * w, 1, 1], dim=2)
            x_prime_map = x_prime.view(B, C, h, w)
            lt_prime_1_1 = lt_prime.view(B, C, 1, 1)
            gt_prime_1_1 = gt_prime.view(B, C, 1, 1)

            w_lt = self._local_token_upsample(lt_prime_1_1, h, w)
            x_weighted = x_prime_map * w_lt

            x_out_quads.append(x_weighted)
            gt_tokens.append(gt_prime)

        top = torch.cat([x_out_quads[0], x_out_quads[1]], dim=3)
        bottom = torch.cat([x_out_quads[2], x_out_quads[3]], dim=3)
        f_wlt = torch.cat([top, bottom], dim=2)

        f_wgt = self._global_token_attention(f_wlt, gt_tokens, H, W)
        f_out = f_wlt + f_wgt

        f_out = self._crop(f_out, pad_hw)
        return f_out


class DTA256(BaseDTA):
    def __init__(self, inplanes: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__(inplanes=inplanes,
                         linear_h_dim=32,
                         linear_H_dim=64,
                         num_heads=num_heads,
                         dropout=dropout)


class DTA512(BaseDTA):
    def __init__(self, inplanes: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__(inplanes=inplanes,
                         linear_h_dim=16,
                         linear_H_dim=32,
                         num_heads=num_heads,
                         dropout=dropout)


class DTA1024(BaseDTA):
    def __init__(self, inplanes: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__(inplanes=inplanes,
                         linear_h_dim=8,
                         linear_H_dim=16,
                         num_heads=num_heads,
                         dropout=dropout)
