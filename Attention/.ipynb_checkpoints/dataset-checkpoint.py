import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SOAPDataset(Dataset):
    def __init__(self, soap_data_list, targets):
        self.soap_data_list = soap_data_list
        self.targets = targets

    def __len__(self):
        return len(self.soap_data_list)

    def __getitem__(self, idx):
        soap_tensor = torch.tensor(self.soap_data_list[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return soap_tensor, target


def mil_collate_fn(batch):
    envs, targets = zip(*batch)
    targets = torch.stack(targets)
    padded_envs = pad_sequence(envs, batch_first=True)

    lengths = torch.tensor([e.shape[0] for e in envs])
    B, K_max = len(envs), padded_envs.shape[1]
    mask = torch.zeros(B, K_max, dtype=torch.float)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    return padded_envs, mask, targets