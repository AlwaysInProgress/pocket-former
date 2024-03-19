from typing import Literal
import tiktoken
import requests
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

small_wikipedia_link = "https://raw.githubusercontent.com/arpytanshu/latent-semantic-indexing/master/wikipedia_utf8_filtered_20pageviews-1K.tsv"
input_file_path = os.path.join(os.path.dirname(__file__), 'data/wiki-data.txt')
train_file_path = os.path.join(os.path.dirname(__file__), 'data/train.bin')
test_file_path = os.path.join(os.path.dirname(__file__), 'data/test.bin')
os.makedirs(os.path.dirname("./data/"), exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
end_of_sentence_token = 50256

def get_fake_data(args):
    # list of tuples of (batch, labels)
    batches = [torch.randn((args.bs, args.seq_len, args.hidden_dim)) for _ in range(10)]
    labels = [torch.randint(0, 50257, (args.bs, args.seq_len)) for _ in range(10)]
    return list(zip(batches, labels))


def preprocess_data():
    print("Preprocessing data")

    with open(input_file_path, 'r') as f:
        data = f.read()

    print("File loaded")

    lines = data.split("\n")

    def process_line(line: str):
        if len(line) == 0:
            return None
        line = line.split(" ", 1)[1].strip()
        tokens = enc.encode(line)
        tokens.append(end_of_sentence_token)
        return tokens

    all_tokens = []
    for line in tqdm(lines):
        line = process_line(line)
        if line is None:
            continue
        all_tokens.extend(line)


    n = len(all_tokens)
    split_percent = 0.9
    train_data = all_tokens[:int(n * split_percent)]
    test_data = all_tokens[int(n * split_percent):]

    print(f"# of train tokens: {len(train_data)}")
    print(f"# of test tokens: {len(test_data)}")

    np.array(train_data, dtype=np.uint16).tofile(train_file_path)
    np.array(test_data, dtype=np.uint16).tofile(test_file_path)

def download_data():
    if not os.path.exists(input_file_path):
        data = requests.get(small_wikipedia_link).text
        with open(input_file_path, 'w') as f:
            f.write(data)

def get_batch(seq_len: int, batch_size: int, split: Literal['train', 'test']):
    file_path = train_file_path if split == 'train' else test_file_path
    data = np.fromfile(file_path, dtype=np.uint16)
    n = len(data)

    idx = np.random.randint(0, n - seq_len, batch_size)

    res = [data[i:i+seq_len] for i in idx]

    return torch.tensor(res, dtype=torch.int64)

def get_epoch(seq_len: int, batch_size: int, epoch_len:int, split: Literal['train', 'test']):
    batches = []
    for _ in range(epoch_len):
        batch = get_batch(seq_len, batch_size, split)
        batches.append((
            # torch.tensor(batch, dtype=torch.int64),
            # torch.tensor(batch, dtype=torch.int64)
            torch.tensor(batch[:-1], dtype=torch.int64), # input
            torch.tensor(batch[1:], dtype=torch.int64), # output
        ))
    return batches

def print_batch(samples: torch.Tensor):
    # Loop through the samples and convert them to text
    for sample in samples:
        print(enc.decode(sample.tolist()))

action = sys.argv[1] if len(sys.argv) > 1 else None

if action == "download":
    download_data()
elif action == "preprocess":
    preprocess_data()
elif action == "sample":
    print("Train:")
    batch = get_batch(20, 50, 'train')
    print_batch(batch)
    print("\nTest:")
    batch = get_batch(20, 50, 'test')
    print_batch(batch)
