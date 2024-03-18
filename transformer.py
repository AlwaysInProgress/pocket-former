from typing import Optional
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from dataset import *
from dataclasses import dataclass

@dataclass
class Args:
    bs: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_epochs: int
    prompt: Optional[str]
    train: bool

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, qkv_dim, dropout=0.):
        super(AttentionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.qkv_dim = qkv_dim
        self.q = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
        self.k = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
        self.v = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)

        self.layer_norm = nn.LayerNorm(self.qkv_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax2d()

    def forward(self, x: torch.Tensor):
        # x is shape (batch_size, seq_len, hidden_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        kT = k.reshape(k.shape[0], k.shape[2], k.shape[1])

        attn_map = self.softmax(q @ kT / self.qkv_dim ** 0.5)
        attn_map = self.dropout(attn_map)
        attn_mask = torch.tril(torch.ones_like(attn_map))
        attn_map_masked = attn_map * attn_mask
        attn_map_masked[attn_mask == 0] = 1e-9

        res = attn_map_masked @ v

        return res

class Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_heads=1, dropout=0., train=False, vocab_size=50257):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        assert self.hidden_dim % self.num_heads == 0
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.attention_heads = nn.ModuleList(
            [AttentionHead(hidden_dim=self.hidden_dim, qkv_dim=self.hidden_dim // self.num_heads) for h in range(num_heads)]
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.softmax = nn.Softmax2d()
        self.dropout = nn.Dropout(p=dropout)
        self.mha_proj = nn.Linear(hidden_dim, hidden_dim)
    
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
            self.dropout
        )

        self.lm_head = nn.Linear(self.hidden_dim, vocab_size)

    def positional_encoding(self, x):
        pass

    def mha(self, x):
        # projections have shape (batch_size, num_heads, seq_len, hidden_dim)
        return self.mha_proj(torch.concatenate([head(x) for head in self.attention_heads], dim=-1))

    def forward(self, x):
        x = self.embedding(x)
        attn = self.mha(x)
        res = self.dropout(x + self.layer_norm(attn))
        res = self.mlp(res) + self.layer_norm(res)
        return self.lm_head(res)

def train_epoch(model, optimizer, args):
    train_loader = get_epoch(args.seq_len, args.bs, epoch_len=10, split='train')
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(batch)
        loss = nn.functional.cross_entropy(output.view((-1, 50257)), labels.view(-1))
        loss.backward()
        optimizer.step()
        loss_sum += loss
    loss_avg = loss_sum / len(train_loader)

    return loss_avg

def eval_epoch(model, args):
    val_loader = get_epoch(args.seq_len, args.bs, epoch_len=10, split='test')
    model.train()
    loss_sum = 0

    with torch.no_grad():
        for batch, labels in tqdm(val_loader):
            output = model(batch)
            loss = nn.functional.cross_entropy(output.view((-1, 50257)), labels.view(-1))
            loss_sum += loss

    loss_avg = loss_sum / len(val_loader)

    return loss_avg    


def train(t: Transformer, args: Args):
    optimizer = torch.optim.AdamW(t.parameters(), lr=1e-3, weight_decay=1e-1)
    t.train()
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(t, optimizer, args)
        print(f'train loss for epoch {epoch}: {train_loss}')
        val_loss = train_epoch(t, optimizer, args)
        print(f'val loss for epoch {epoch}: {val_loss}')

def prompt(t: Transformer, prompt: str) -> str:
    # t.eval()
    #
    # enc = tiktoken.get_encoding("gpt2")
    # encoded_p = enc.encode(prompt)
    #
    # eos = torch.zeros((seq_len)) # eos token
    # x = torch.tensor(encoded_p)
    # max_tokens = 10
    #
    # with torch.no_grad():
    #     for token in range(max_tokens):
    #         probs = t(x)[-1]
    #         next_token = torch.argmax(probs, dim=-1)
    #         if next_token == eos:
    #             break
    #         x = torch.concatenate([x[1:], next_token], dim=0)

    return "Hello World"


# define custom args type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)    
    parser.add_argument("--hidden_dim", type=int, default=128)    
    parser.add_argument("--num_heads", type=int, default=2)    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    # Casts args to dataclass
    args = Args(
        bs=args.bs,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        train=args.train,
        prompt=args.prompt
    );

    t = Transformer(
        seq_len=args.seq_len, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        train=False
    )


    if args.train:
        train(t, args)
    elif not args.prompt is None:
        print("Prompt: ", args.prompt)
        res = prompt(t, args.prompt)
        print("Response:", res)
        exit()
    else:
        parser.print_help()

