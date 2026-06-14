import torch
import torch.nn as nn


class FeatureEmbedder(nn.Module):
    def __init__(self, in_dim, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3, ffn_mult=4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        hidden_dim = ffn_mult * d_model
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, context, key_padding_mask=None):
        qkv = self.attn_norm(context)
        attn_out, _ = self.mha(
            qkv, qkv, qkv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = context + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3, ffn_mult=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        hidden_dim = ffn_mult * d_model
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, context, key_padding_mask=None):
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        
        attn_out, _ = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = queries + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class CondNet(nn.Module):
    # TODO: rethink abt is it proper to use HL(y)
    def __init__(self, in_channels, out_channels, c_dim, d_model=128, nhead=8, dropout=0.2, depth=3):
        super().__init__()
        expected_c_dim = 13
        if c_dim != expected_c_dim:
            raise ValueError(
                f"CondNet expects c_dim={expected_c_dim} from load_data.py "
                f"[met(2), jet0(4), jet1(4), dilepton HL(3)], got {c_dim}."
            )
        if in_channels <= c_dim:
            raise ValueError(f"in_channels ({in_channels}) must be > c_dim ({c_dim})")
        if out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even, got {out_channels}")

        self.c_dim = c_dim
        # 1. Embedders
        self.inn_embed = FeatureEmbedder(in_channels - c_dim, d_model, dropout)
        self.met_embed = FeatureEmbedder(2, d_model, dropout)
        self.jet_embed = FeatureEmbedder(4, d_model, dropout)
        self.lep_hl_embed = FeatureEmbedder(3, d_model, dropout)
        self.num_tokens = 4  # [met, jet0, jet1, lep_hl]
        self.token_embed = nn.Parameter(torch.empty(self.num_tokens, d_model))
        nn.init.normal_(self.token_embed, std=0.02)
        
        # 2. MHA stack: condition tokens interact, then the INN token queries them.
        self.input_refiner_lst = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, dropout=dropout) for _ in range(depth)
        ])
        self.context_refiner_lst = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout=dropout) for _ in range(depth)
        ])
        
        # 3. Output Head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, out_channels)
        )
        nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_head[-1].bias)

        self.query_skip = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # Expected x: [inn_half(..), met(2), jet0(4), jet1(4), lep_hl(3)]
        x_inn = x[:, :-self.c_dim]
        x_cond = x[:, -self.c_dim:]

        # --- Step 1: Embedding MLP ---
        # observable tokens
        met = self.met_embed(x_cond[:, 0:2])
        j0 = self.jet_embed(x_cond[:, 2:6])
        j1 = self.jet_embed(x_cond[:, 6:10])
        lep_hl = self.lep_hl_embed(x_cond[:, 10:])
        # intermediate inn token
        inn_tok = self.inn_embed(x_inn)
        
        # context tokens are from condition only: [met, jet0, jet1, lep_hl]
        context = torch.stack([met, j0, j1, lep_hl], dim=1)
        context = context + self.token_embed.unsqueeze(0)
        # context = torch.stack([met, j0, j1, j2], dim=1)
        # query token is from INN input
        inn_query = inn_tok.unsqueeze(1)  # [B, 1, d_model]
        inn_query = inn_query + self.query_skip(inn_query)
        
        batch_size = x.shape[0]
        key_mask = torch.zeros((batch_size, context.shape[1]), dtype=torch.bool, device=x_cond.device)
        # Mask jet tokens that are exactly zero 4-vectors.
        key_mask[:, 1] = (torch.abs(x_cond[:, 2:6]).sum(dim=1) == 0.0)  # Jet 0 token
        key_mask[:, 2] = (torch.abs(x_cond[:, 6:10]).sum(dim=1) == 0.0)  # Jet 1 token

        # --- Step 2: SA and CA ---
        for input_refiner, context_refiner in zip(self.input_refiner_lst, self.context_refiner_lst):
            context = input_refiner(context, key_padding_mask=key_mask)
            # Keep padded jet slots from leaking signal through residual paths.
            context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
            inn_query = context_refiner(inn_query, context, key_padding_mask=key_mask)
            
        # --- Step 3: Output Head ---
        # only take out refined inn token
        outputs = self.output_head(inn_query.squeeze(1))
        
        return outputs
