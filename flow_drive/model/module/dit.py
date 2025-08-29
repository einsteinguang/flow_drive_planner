import math
import torch
import torch.nn as nn


def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, cross_c, t, key_padding_mask):
        # x: (B, P, D) - input sequence
        # cross_c: (B, N, D) - cross attention conditioning
        # t: (B, D) - timestep embedding
        # key_padding_mask: (B, N) - mask for cross attention

        mask = ~key_padding_mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=1).clamp(min=1)
        mean_valid = (cross_c * mask).sum(dim=1) / valid_counts  # (B, D)
        t_final = t + mean_valid  # Add mean of valid cross_c to timestep embedding

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_final).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c, key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp2(self.norm4(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(self, n_blocks, action_dim, pred_horizon, hidden_dim=256, heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.preproj = nn.Sequential(
            nn.Linear(action_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, pred_horizon, hidden_dim))
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(
            hidden_dim, heads, dropout, mlp_ratio) for i in range(n_blocks)])
        self.final_layer = FinalLayer(hidden_dim, action_dim)

        total = sum(p.numel() for p in self.parameters())
        # print(f"Number of dit parameters: {total:e}")
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, sample, timestep, global_cond):
        """
        Forward pass of DiT.
        x: (B, P, action_dim)   -> Actions
        t: (B,)
        cond: {"encoding": (B, N, D), "mask": (B, N, N)} -> conditional information
        """
        B, P, _ = sample.shape

        x = self.preproj(sample)  # (B, P, D)
        # add positional embedding
        x = x + self.pos_emb[:, :P, :]  # (B, P, D)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0]).view(B)
        time_emb = self.t_embedder(timesteps)  # (B, D)

        for block in self.blocks:
            x = block(x, global_cond["encoding"], time_emb, global_cond["mask"])

        x = self.final_layer(x, time_emb)
        return x
