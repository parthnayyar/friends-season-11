import torch

class DataLoader():
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def get_batch(self, split, block_size, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        data = self.train if split == 'train' else self.test
        ix = torch.randint(len(data) - block_size, (batch_size, ))
        x = torch.stack([data[i: i+block_size] for i in ix])
        y = torch.stack([data[i+1: i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y