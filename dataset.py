from typing import Literal
import tiktoken
import requests
import os
import sys
import numpy as np
import torch

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

def downloadData():
    if not os.path.exists(input_file_path):
        data = requests.get(small_wikipedia_link).text
        with open(input_file_path, 'w') as f:
            f.write(data)

    with open(input_file_path, 'r') as f:
        data = f.read()

    lines = data.split("\n")

    lines = [x for x in lines if len(x) > 0]
    lines = [line.split(" ", 1)[1].strip() for line in lines] # Remove unwanted starting tokens
    lines = [enc.encode(x) for x in lines]
    for line in lines:
        line.append(end_of_sentence_token)
    lines = [item for sublist in lines for item in sublist]

    n = len(lines)
    split_percent = 0.9
    train_data = lines[:int(n * split_percent)]
    test_data = lines[int(n * split_percent):]

    np.array(train_data, dtype=np.uint16).tofile(train_file_path)
    np.array(test_data, dtype=np.uint16).tofile(test_file_path)

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

download_flag = sys.argv[1] == "download" if len(sys.argv) > 1 else False
sample_flag = sys.argv[1] == "sample" if len(sys.argv) > 1 else False

if download_flag:
    downloadData()

if sample_flag:
    print("Train:")
    batch = get_batch(20, 50, 'train')
    print_batch(batch)
    print("\nTest:")
    batch = get_batch(20, 50, 'test')
    print_batch(batch)
