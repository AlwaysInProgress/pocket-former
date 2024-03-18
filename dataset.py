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

    np.array(train_data, dtype=np.int16).tofile(train_file_path)
    np.array(test_data, dtype=np.int16).tofile(test_file_path)


def get_batch(seq_len: int, batch_size: int, split: Literal['train', 'test']):
    file_path = train_file_path if split == 'train' else test_file_path
    data = np.fromfile(file_path, dtype=np.int16)
    n = len(data)

    idx = np.random.randint(0, n - seq_len, batch_size)

    return torch.tensor([data[i:i+seq_len] for i in idx])

def get_train_loader(num_batches: int, seq_len: int, batch_size: int):
    for _ in range(num_batches):
        return (torch.tensor(get_batch(seq_len, batch_size, 'train')), torch.tensor(get_batch(seq_len, batch_size, 'train')))

def get_val_loader(num_batches: int, seq_len: int, batch_size: int):
    for _ in range(num_batches):
        return (torch.tensor(get_batch(seq_len, batch_size, 'test')), torch.tensor(get_batch(seq_len, batch_size, 'test')))

def print_batch(samples):
    samples = [enc.decode(x.tolist()) for x in samples]
    for sample in samples:
        print(sample)


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