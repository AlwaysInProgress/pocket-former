from typing import Optional
import torch
from torch._prims_common import check
import torch.nn as nn
import argparse
from tqdm import tqdm
from dataset import *
import datetime
from dataclasses import dataclass
import math

@dataclass
class Args:
    bs: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_epochs: int
    prompt: Optional[str]
    train: bool
    checkpoint: Optional[str]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, hidden_dim, seq_len, num_heads=1, dropout=0., num_blocks=6, train=False, vocab_size=50257):
        super(Transformer, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        assert self.hidden_dim % self.num_heads == 0
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim).to(device)
        self.positional_encoding = nn.Parameter(torch.randn((self.seq_len, self.hidden_dim))).to(device) 
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.softmax = nn.Softmax2d()
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
                nn.ReLU(),
                nn.Linear(4*self.hidden_dim, self.hidden_dim),
                nn.Dropout(dropout)
            )
            block = nn.ModuleList([attention_heads, mha_projection, mlp]).to(device)
            self.blocks.append(block)

        self.to(device)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

    def mha(self, x):
        # projections have shape (batch_size, num_heads, seq_len, hidden_dim)
        return self.mha_proj(torch.concatenate([head(x) for head in self.attention_heads], dim=-1))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for block in self.blocks:
            attn = torch.cat([head(x) for head in block[0]], dim=-1) # mha
            attn = block[1](attn) # mha_proj
            attn = self.dropout(x + self.layer_norm(attn)) # add and norm
            res = block[2](attn) + self.layer_norm(attn) # mlp
            x = res
        return self.lm_head(res)

def train_epoch(model, optimizer, args, train_dataset):
    train_loader = get_epoch(
        args.seq_len, 
        args.bs, 
        epoch_len=1000, 
        dataset=train_dataset
    )
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.view(-1).to(device)
        output = model(batch).view((-1, model.vocab_size))
        probs = torch.softmax(output, dim=-1)
        loss = nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    loss_avg = loss_sum / len(train_loader)

    return loss_avg

def eval_epoch(model: Transformer, args: Args, test_dataset: np.ndarray) -> float:
    val_loader = get_epoch(
        args.seq_len, 
        args.bs, 
        epoch_len=10, 
        dataset=test_dataset
    )
    model.train()
    loss_sum = 0

    with torch.no_grad():
        for batch, labels in tqdm(val_loader):
            batch = batch.to(device)
            labels = labels.view(-1).to(device)
            output = model(batch).view((-1, model.vocab_size))
            loss = nn.functional.cross_entropy(output, labels)
            loss_sum += loss.item()

    loss_avg = loss_sum / len(val_loader)

    return loss_avg    


def train(model: Transformer, args: Args, train_dataset: np.ndarray, test_dataset: np.ndarray):
    optimizer = torch.optim.AdamW(t.parameters(), lr=1e-4, weight_decay=0) # 1e-1)
    t.train()
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, optimizer, args, train_dataset)
        val_loss = train_epoch(t, optimizer, args, test_dataset)
        print(f'train loss for epoch {epoch}: {train_loss}')
        print(f'train perplexity: {math.exp(train_loss)}')
        print(f'val loss for epoch {epoch}: {val_loss}')
        print(f'val perplexity: {math.exp(val_loss)}')
        save_model(model, "checkpoints")

def save_model(model, path, name=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        name = f"model_{datetime_str}"

    torch.save(model.state_dict(), os.path.join(path, name) + ".pt")
    print(f"Model saved to {os.path.join(path, name)} + .pt")

def pad_sequence(seq: torch.Tensor, seq_len: int, device: torch.device, pad_index: int = 0) -> torch.Tensor:
    # seq has shape (bs, seq_len)
    if seq.shape[1] < seq_len:
        return torch.cat([seq, torch.ones((seq.shape[0], seq_len - seq.shape[1])).to(device) * pad_index], dim=1)
    else:
        return seq

def prompt(t: Transformer, args: Args) -> str:
    t.eval()
    if args.checkpoint is not None:
        t.load_state_dict(torch.load(args.checkpoint))

    print("Prompt: ", args.prompt)

    enc = tiktoken.get_encoding("gpt2")
    
    if args.prompt is None:
        print("Prompt is None")
        return "Error: prompt is None"

    encoded_p = enc.encode(args.prompt)
    # encoded_p = get_batch(args.seq_len, 1, 'test').squeeze().tolist()
    print("prompt decoded: ", enc.decode(encoded_p))
    # eos = torch.zeros((seq_len)) # eos token
    x = torch.tensor(encoded_p).unsqueeze(0).to(device)
    max_gen_tokens = 6
    
    with torch.no_grad():
        for token in range(max_gen_tokens):
            padded_x = pad_sequence(x, args.seq_len, device).long()
            logits = t(padded_x).transpose(0, 1)[-1]
            probs = torch.softmax(logits, dim=-1)
            # next_token = torch.argmax(probs, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, 1)[0]
            if next_token == 50256:
                break

            if x.shape[1] == args.seq_len:
                x = x[:, 1:]

            x = torch.concatenate([x, next_token.unsqueeze(0)], dim=1)

    return enc.decode(x.squeeze().tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)    
    parser.add_argument("--hidden_dim", type=int, default=128)    
    parser.add_argument("--num_heads", type=int, default=2)    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()


    # Casts args to dataclass
    args = Args(
        bs=args.bs,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        train=args.train,
        prompt=args.prompt,
        checkpoint=args.checkpoint
    )

    t = Transformer(
        seq_len=args.seq_len, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads,
        num_blocks=6,
        train=False
    )
    t.to(device)

    train_dataset = get_data('train')
    test_dataset = get_data('test')

    if args.train:
        train(t, args, train_dataset, test_dataset)
    elif not args.prompt is None:
        res = prompt(t, args)
        print("Response:", res)
        exit()
    else:
        parser.print_help()
