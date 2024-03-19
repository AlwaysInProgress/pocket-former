from typing import List, Literal
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
        return line

    print("Processing lines")
    processed_lines: List[str] = []
    for line in tqdm(lines):
        line = process_line(line)
        if line is not None:
            processed_lines.append(line)

    print("Encoding tokens")
    all_tokens = enc.encode_ordinary_batch(processed_lines)

    # Add end of sentence token
    print("Adding end of sentence token")
    all_tokens = [token + [end_of_sentence_token] for token in all_tokens]

    # Flatten the list
    all_tokens = [token for sublist in all_tokens for token in sublist]

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

def get_data(split: Literal['train', 'test']):
    file_path = train_file_path if split == 'train' else test_file_path
    return np.fromfile(file_path, dtype=np.uint16)

def get_batch(
    seq_len: int, 
    batch_size: int, 
    dataset: np.ndarray,
):
    n = len(dataset)
    idx = np.random.randint(0, n - seq_len, batch_size)
    res = [dataset[i:i+seq_len] for i in idx]
    return torch.tensor(res, dtype=torch.int64)

def get_epoch(
    seq_len: int, 
    batch_size: int, 
    epoch_len:int, 
    dataset: np.ndarray,
):
    batches = []
    for _ in range(epoch_len):
        batch = get_batch(seq_len, batch_size, dataset)
        batches.append((
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
    data = get_data('train')
    batch = get_batch(20, 50, data)
    print_batch(batch)
    print("\nTest:")
    data = get_data('test')
    batch = get_batch(20, 50, data)
    print_batch(batch)
