import tiktoken
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

enc = tiktoken.get_encoding("cl100k_base")

class CustomDataset(Dataset):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        with open('data.txt', 'r') as f:
            self.data = [enc.encode(x) for x in f.readlines()]

        self.sample_counts = [len(line) - self.seq_len for line in self.data]
        self.len = sum(self.sample_counts)

        self.hist = [0]
        for i, sample_count in enumerate(self.sample_counts):
            self.hist.append(self.hist[i] + sample_count)


    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):

        line_idx = 0
        for i in range(len(self.hist)):
            if self.hist[i] > idx:
                line_idx = i - 1
                break

        line = self.data[line_idx]
        start = idx - self.hist[line_idx]
        return line[start:start+self.seq_len]



def test():
    dataset = CustomDataset(8)
    data_loader = DataLoader(dataset, batch_size=100)
    sample = next(iter(data_loader))
    # Decode the sample
    for s in sample:
        print(enc.decode(s.tolist()))


test()

