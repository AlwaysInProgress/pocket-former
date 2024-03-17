from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        with open('data.txt', 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        return line


