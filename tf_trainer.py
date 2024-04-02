import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from dataset import *
import gc 
from tf_utils import *
from transformer import Transformer
from typing import Optional
import argparse
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
    num_blocks: int
    epoch_len: int
    use_warmup_cos_decay: bool
    clip_grad: float

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model: Transformer, optimizer: torch.optim.Optimizer, args: Args, train_dataset: np.ndarray, global_iter: int = 0) -> float:
    train_loader = get_epoch(
        args.seq_len, 
        args.bs, 
        epoch_len=args.epoch_len, 
        dataset=train_dataset
    )
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        if args.use_warmup_cos_decay:
            # lr = get_lin_warmup_cos_decay_lr(it=global_iter, warmup_iters=2000, lr_decay_iters=600000, min_lr=6e-5, max_lr=6e-4)
            lr = get_lin_warmup_cos_decay_lr(it=global_iter, warmup_iters=2000, lr_decay_iters=args.num_epochs*args.epoch_len, min_lr=6e-5, max_lr=6e-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        batch = batch.to(device)
        labels = labels.view(-1).to(device)
        optimizer.zero_grad()
        output = model(batch).view((-1, model.vocab_size))
        probs = torch.softmax(output, dim=-1)
        loss = nn.functional.cross_entropy(output, labels)
        if torch.isfinite(loss):
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
        loss_sum += loss.item()

        # free up memory
        del batch, labels, output, loss
        gc.collect()

        global_iter += 1

    loss_avg = loss_sum / len(train_loader)

    # free up memory
    del train_loader
    gc.collect()

    return loss_avg

def eval_epoch(model: Transformer, args: Args, test_dataset: np.ndarray, epoch_len: int = 100) -> float:
    val_loader = get_epoch(
        args.seq_len, 
        args.bs, 
        epoch_len=epoch_len, 
        dataset=test_dataset
    )
    model.eval()
    loss_sum = 0

    with torch.no_grad():
        for batch, labels in tqdm(val_loader):
            batch = batch.to(device)
            labels = labels.view(-1).to(device)
            output = model(batch).view((-1, model.vocab_size))
            loss = nn.functional.cross_entropy(output, labels)
            loss_sum += loss.item()

            # free up memory
            del batch, labels, output, loss
            gc.collect()

    loss_avg = loss_sum / len(val_loader)

    # free up memory
    del val_loader
    gc.collect()

    return loss_avg   
     

def train(model: Transformer, args: Args, train_dataset: np.ndarray, test_dataset: np.ndarray, epoch_len: int = 1000, use_warmup_cos_decay: bool = False, global_iter: int = 0):
    # optimizer = torch.optim.AdamW(t.parameters(), lr=2.5e-4, weight_decay=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    t.train()
    for epoch in range(args.num_epochs):
        print(f"Training Epoch {epoch + 1}...")
        train_loss = train_epoch(model, optimizer, args, train_dataset, global_iter=global_iter)
        print(f"Evaluating Epoch {epoch + 1}...")
        val_loss = eval_epoch(t, args, test_dataset, epoch_len=100)
        tokens_seen = (epoch + 1) * args.bs * args.seq_len * epoch_len
        print(f'train loss for epoch {epoch + 1}: {train_loss}')
        print(f'train perplexity: {math.exp(train_loss)}')
        print(f'val loss for epoch {epoch + 1}: {val_loss}')
        print(f'val perplexity: {math.exp(val_loss)}')
        print(f'total tokens seen: {tokens_seen}')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'global_iter': global_iter,
            'args': args
        }

        save_model(checkpoint, "checkpoints", f"epoch_{epoch + 1}_{tokens_seen}_tokens_ppl_{round(math.exp(val_loss), 2)}_")

def prompt(t: Transformer, args: Args, checkpoint: str = None, dataset: np.ndarray = None):
    t.eval()
    if args.checkpoint is not None:
        # t.load_state_dict(torch.load(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        saved_model = checkpoint['model_state_dict']
        t.load_state_dict(saved_model)

    enc = tiktoken.get_encoding("gpt2")
    # encoded_p = get_batch(seq_len=args.seq_len, batch_size=1, dataset=dataset)
    encoded_p = torch.tensor(enc.encode(args.prompt)).unsqueeze(0)
    print("Prompt: ", enc.decode(encoded_p.squeeze().tolist()))
    # print("Prompt: ", enc.decode(encoded_p))
    
    if args.prompt is None:
        print("Prompt is None")
        return "Error: prompt is None"

    # eos = torch.zeros((seq_len)) # eos token
    x = torch.tensor(encoded_p).to(device)
    max_gen_tokens = 10
    
    with torch.no_grad():
        for token in range(max_gen_tokens):
            padded_x = pad_sequence(x, args.seq_len, device).long()
            logits = t(padded_x).transpose(0, 1)[-1]
            logits = top_k_top_p_filtering(logits, top_k=40)
            probs = torch.softmax(logits, dim=-1)
            # next_token = torch.argmax(probs, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, 1)[0]
            # next_token = nucleus_sampling(probs, p=0.9)
            if next_token == 50256:
                break

            if x.shape[1] == args.seq_len:
                x = x[:, 1:]

            x = torch.concatenate([x, next_token.unsqueeze(0)], dim=1)
            print(enc.decode(x.squeeze().tolist()))

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
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--use_warmup_cos_decay", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=1.)
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
        checkpoint=args.checkpoint,
        num_blocks=args.num_blocks,
        epoch_len=args.epoch_len,
        use_warmup_cos_decay=args.use_warmup_cos_decay,
        clip_grad=args.clip_grad
    )

    t = Transformer(
        hidden_dim=args.hidden_dim, 
        seq_len=args.seq_len, 
        num_heads=args.num_heads,
        dropout=0.0,
        num_blocks=args.num_blocks,
        train=False
    )
    t.to(device)
    print(t)

    train_dataset = get_data('train')
    test_dataset = get_data('test')

    global_iter = 0
    if args.train:
        train(t, args, train_dataset, test_dataset, epoch_len=args.epoch_len, use_warmup_cos_decay=args.use_warmup_cos_decay, global_iter=global_iter)
    elif args.prompt is not None:
        res = prompt(t=t, args=args, dataset=test_dataset)
        print("Response:", res)
        exit()
    else:
        parser.print_help()