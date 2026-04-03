import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, context, key_padding_mask=None):
        res = context
        qkv = self.norm(context)
        attn_out, _ = self.mha(qkv, qkv, qkv, key_padding_mask=key_padding_mask)
        x = res + attn_out
        
        x = x + self.ffn(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, queries, context, key_padding_mask=None):
        res = queries
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        
        attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask)
        x = res + attn_out
        
        x = x + self.ffn(x)
        return x

class CondNet(nn.Module):
    # TODO: rethink abt is it proper to use HL(y)
    def __init__(self, in_channels, out_channels, c_dim, d_model=64, nhead=8, dropout=0.5):
        super().__init__()
        if in_channels <= c_dim:
            raise ValueError(f"in_channels ({in_channels}) must be > c_dim ({c_dim})")
        if out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even, got {out_channels}")

        self.c_dim = c_dim
        # 1. Embedders
        self.inn_embed = nn.Linear(in_channels - c_dim, d_model)
        self.met_embed = nn.Linear(2, d_model)
        self.jet_embed = nn.Linear(4, d_model)
        self.ang_embed = nn.Linear(5, d_model)
        self.num_tokens = 5  # [met, jet0, jet1, jet2, ang]
        
        # 2. Lightweight MHA: inn token queries condition tokens.
        self.input_refiner_lst = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout=dropout) for _ in range(2)])
        self.context_refiner_lst = nn.ModuleList([CrossAttentionBlock(d_model, nhead, dropout=dropout) for _ in range(2)])
        
        # 3. Output Head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels)
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x):
        # Expected x: [inn_half(..), jet0(4), jet1(4), jet2(4), ang(6)]
        x_inn = x[:, :-self.c_dim]
        x_cond = x[:, -self.c_dim:]

        # --- Step 1: Embedding MLP ---
        # observable tokens
        met = self.met_embed(x_cond[:, :2])
        j0 = self.jet_embed(x_cond[:, 2:6])
        j1 = self.jet_embed(x_cond[:, 6:10])
        j2 = self.jet_embed(x_cond[:, 10:14])
        ang = self.ang_embed(x_cond[:, 14:])
        # intermediate inn token
        inn_tok = self.inn_embed(x_inn)
        
        # context tokens are from condition only: [jet0, jet1, jet2, ang]
        context = torch.stack([met, j0, j1, j2, ang], dim=1) # TODO: position info (!)
        # context = torch.stack([met, j0, j1, j2], dim=1)
        # query token is from INN input
        inn_query = inn_tok.unsqueeze(1)  # [B, 1, d_model]
        
        batch_size = x.shape[0]
        key_mask = torch.zeros((batch_size, context.shape[1]), dtype=torch.bool, device=x_cond.device)
        # Mask jet tokens that are exactly zero 4-vectors.
        key_mask[:, 1] = (torch.abs(x_cond[:, 2:6]).sum(dim=1) == 0)  # Jet 0 token
        key_mask[:, 2] = (torch.abs(x_cond[:, 6:10]).sum(dim=1) == 0)  # Jet 1 token
        key_mask[:, 3] = (torch.abs(x_cond[:, 10:14]).sum(dim=1) == 0) # Jet 2 token

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
