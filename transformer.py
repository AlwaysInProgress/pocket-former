import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, qkv_dim, dropout=0.1):
        super(AttentionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.q = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
        self.k = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
        self.v = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)

        self.layer_norm = nn.LayerNorm(self.qkv_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # x is shape (batch_size, seq_len, hidden_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        kT = k.reshape(k.shape[0], k.shape[2], k.shape[1])

        attn_map = q @ kT / self.qkv_dim ** 0.5

        # attn_mask = torch.tril(torch.ones_like(attn_map))
        # attn_map_masked = torch.where(attn_mask == 0, -1e9, attn_map)
        # subtracting 1e9 instead
        attn_map_masked = attn_map - 1e9 * (1 - torch.tril(torch.ones_like(attn_map)))
        attn_map = self.softmax(attn_map_masked)
        attn_map = self.dropout(attn_map_masked)

        res = attn_map_masked @ v

        return res

class Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_heads=1, dropout=0.1, num_blocks=6, train=False, vocab_size=50257):
        super(Transformer, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        assert self.hidden_dim % self.num_heads == 0
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim).to(device)
        self.positional_encoding = nn.Parameter(torch.randn((self.seq_len, self.hidden_dim))).to(device) 
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.mha_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)
    
        # transformer blocks with my mha and mlp
        self.blocks = nn.ModuleList() 
        for _ in range(self.num_blocks):
            attention_heads = nn.ModuleList([AttentionHead(self.hidden_dim, self.hidden_dim // self.num_heads, dropout) for _ in range(self.num_heads)])
            mha_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, 4*self.hidden_dim),
                # nn.ReLU(),
                nn.GELU(),
                nn.Linear(4*self.hidden_dim, self.hidden_dim),
                nn.Dropout(dropout)
            )
            block = nn.ModuleList([attention_heads, mha_projection, mlp]).to(device)
            self.blocks.append(block)
        
        self.to(device)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

    # def mha(self, x):
    #     # projections have shape (batch_size, num_heads, seq_len, hidden_dim)
    #     return self.mha_proj(torch.concatenate([head(x) for head in self.attention_heads], dim=-1))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for block in self.blocks:
            attn = torch.cat([head(x) for head in block[0]], dim=-1) # mha
            attn = block[1](attn) # mha_proj
            attn = self.dropout(x + self.layer_norm(attn)) # norm and add
            res = attn + self.layer_norm(block[2](attn)) # mlp
            x = res
        res = self.layer_norm(x)
        return self.lm_head(res)
