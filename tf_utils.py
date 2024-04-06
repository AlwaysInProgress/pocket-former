import torch
import datetime
from torch import Tensor
import torch.nn.functional as F
import os
import math 

def save_model(checkpoint, path, name=""):
    if not os.path.exists(path):
        os.makedirs(path)
    
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = name + datetime_str + '.pt'
    print(f"Saving model to {os.path.join(path, name)}")
    torch.save(checkpoint, os.path.join(path, name))

def pad_sequence(seq: torch.Tensor, seq_len: int, device: torch.device, pad_index: int = 0) -> torch.Tensor:
    # seq has shape (bs, seq_len)
    if seq.shape[1] < seq_len:
        return torch.cat([seq, torch.ones((seq.shape[0], seq_len - seq.shape[1])).to(device) * pad_index], dim=1)
    else:
        return seq

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Adapted from: https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html#top_k_top_p_filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def get_lin_warmup_cos_decay_lr(it, warmup_iters, lr_decay_iters, min_lr, max_lr):
    """
    learning rate decay scheduler (cosine with warmup).
    Adapted from https://github.com/karpathy/nanoGPT/blob/master/train.py#L230C1-L243C1
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters # linear from 0 to lr
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) if between warmup and decay iters, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters) # this is the fraction of the way through decay iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # this expression goes from 1 to 0 with a cos shape
    return min_lr + coeff * (max_lr - min_lr) # interpolate between max and min lr with cos shape