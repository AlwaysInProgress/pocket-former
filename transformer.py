import torch
import torch.nn as nn
import argparse
import tiktoken
from tqdm import tqdm

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
        attn_map_masked[attn_mask == 0] = -torch.inf

        print("attn map: ", attn_map_masked)

        res = attn_map_masked @ v

        return res

class Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_heads=1, dropout=0., train=False):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        assert self.hidden_dim % self.num_heads == 0
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

    def mha(self, x):
        # projections have shape (batch_size, num_heads, seq_len, hidden_dim)
        return self.mha_proj(torch.concatenate([head(x) for head in self.attention_heads], dim=-1))

    def forward(self, x):
        attn = self.mha(x)
        res = self.dropout(x + self.layer_norm(attn))
        return self.mlp(res) + self.layer_norm(res)
        
def get_fake_data(args):
    # return torch.zeros((args.bs, args.seq_len, args.hidden_dim))
    return torch.randn((args.bs, args.seq_len, args.hidden_dim))

def train_epoch(model, train_loader, optimizer):
    model.train()
    loss_sum = 0

    for batch, labels in train_loader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(model(batch), labels)
        loss.backward()
        optimizer.step()
        loss += loss_sum

    loss_avg = loss_sum / len(train_loader)

    return loss_avg  

def eval_epoch(model, val_loader):
    model.train()
    loss_sum = 0

    with torch.no_grad():
        for batch, labels in val_loader:
            loss = nn.CrossEntropyLoss(model(batch), labels)
            loss += loss_sum

    loss_avg = loss_sum / len(val_loader)

    return loss_avg    

def train(model, train_loader, val_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f'train loss for epoch {epoch}: {train_loss}')
        val_loss = train_epoch(model, val_loader, optimizer)
        print(f'val loss for epoch {epoch}: {val_loss}')

def prompt(prompt: str, bs: int, seq_len: int) -> str:
    t = Transformer()
    t.eval()

    enc = tiktoken.get_encoding("cl100k_base")
    encoded_p = enc.encode(prompt)

    eos = torch.zeros((seq_len, bs)) # eos token
    x = torch.tensor(encoded_p)
    max_tokens = 10

    with torch.no_grad():
        for token in range(max_tokens):
            probs = t(x)[-1]
            next_token = torch.argmax(probs, dim=-1)
            if next_token == eos:
                break
            x = torch.concatenate([x[1:], next_token], dim=0)

    response = enc.decode(x)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)    
    parser.add_argument("--hidden_dim", type=int, default=128)    
    parser.add_argument("--num_heads", type=int, default=2)    
    parser.add_argument("--prompt", type=str, default="How do you want to die?")
    args = parser.parse_args()

    mode = "train" if args.train else "prompt"

    # if args.train:
    #     train()
    # else:
    #     prompt(args.prompt, args.bs, args.seq_len)

    t = Transformer(seq_len=args.seq_len, hidden_dim=args.hidden_dim, num_heads=args.num_heads, train=False)
    t.eval()
    data = get_fake_data(args)

    output = t(data)
    print("output shape", output.shape)
    print("output", output)