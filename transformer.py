import torch
import torch.nn as nn
import argparse
import tiktoken

class Transformer(nn.Module):
    def __init__(self, hidden_dim, train=False):
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.mlp = nn.ModuleList([
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
        ])
    
    
    def attention(self, x: torch.Tensor):
        # x is shape (batch_size, seq_length, hidden_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        kT = k.reshape(k.shape(0), k.shape(2), k.shape(1))

        res = (q @ kT / torch.sqrt(self.hidden_dim)) @ v

        return res

    def mha(self, x):
        pass

    def forward(self, x):
        attn = self.attn(x)
        # TODO: add dropout
        res = x + nn.LayerNorm(attn)
        return self.mlp(res) + nn.LayerNorm(res)
        

def get_data(bs=32, seq_len=8, hidden_dim=1024):
    return torch.zeros((bs, seq_len, hidden_dim))

def train():
    pass

def prompt(prompt: str, bs: int, seq_len: int) -> str:
    t = Transformer()
    t.eval()

    enc = tiktoken.get_encoding("cl100k_base")
    encoded_p = enc.encode(prompt)

    # x = get_data()
    eos = torch.zeros((seq_len, bs)) # eos token
    x = torch.tensor(encoded_p)
    max_tokens = 10

    with torch.no_grad():
        for token in range(max_tokens):
            probs = t(x)[-1]
            next_token = torch.argmax(probs, dim=-1)
            if next_token == eos:
                break
            x = torch.concatenate(x[1:], next_token, dim=0)

    response = enc.decode(x)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)    
    parser.add_argument("--prompt", type=str, default="How do you want to die?")
    args = parser.parse_args()

    mode = "train" if args.train else "prompt"

    if args.train:
        train()
    else:
        prompt(args.prompt, args.bs ,args.seq_len)




    
