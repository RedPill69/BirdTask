from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy
from sklearn.preprocessing import StandardScaler

class BirdDataset(Dataset):
    def __init__(self, birdset, is_train, device, left_offset, right_offset, scaler=None):
        self.device = device
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.data = []
        self.labels = []
        self.ignored = []
        self.counts = []
        self.is_train = is_train

        for bird in birdset:
            if is_train:
                self.data.extend(birdset[bird]['train_features'])
                self.labels.extend(birdset[bird]['train_labels'])
                for call in birdset[bird]['train_labels']:
                    self.counts.append(len(call))
                    expanded = scipy.ndimage.maximum_filter1d(call, 3)
                    ignore = (expanded != call) & (call == 0)
                    self.ignored.append(ignore)
            else:
                self.data.extend(birdset[bird]['test_features'])
                self.labels.extend(birdset[bird]['test_labels'])
                for call in birdset[bird]['test_labels']:
                    self.counts.append(len(call))
                    expanded = scipy.ndimage.maximum_filter1d(call, 3)
                    ignore = (expanded != call) & (call == 0)
                    self.ignored.append(ignore)

        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(np.concatenate(self.data))
        self.scaler = scaler

        self.length = np.sum(self.counts)
        self.counts = np.cumsum(self.counts)

        self.data.insert(0, np.zeros((left_offset, 548)))
        self.labels.insert(0, np.zeros((left_offset), dtype=int))
        self.ignored.insert(0, np.ones((left_offset), dtype=int))

        self.data.append(np.zeros((right_offset, 548)))
        self.labels.append(np.zeros((right_offset), dtype=int))
        self.ignored.append(np.ones((right_offset), dtype=int))

        onehot = np.eye(7)
        for i, (call, label, ignored) in enumerate(zip(self.data, self.labels, self.ignored)):
            self.data[i] = torch.tensor(self.scaler.transform(call), dtype=torch.float32).to(self.device)
            self.labels[i] = torch.tensor(onehot[label]).to(self.device)
            self.ignored[i] = torch.tensor(ignored).to(self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        call_index = np.searchsorted(self.counts, index+1) + 1
        call = self.data[call_index]
        target_index = index - (self.counts[call_index-1] - len(call))
        y = self.labels[call_index][target_index]
        ignored = self.ignored[call_index][target_index]

        left_call_index = np.random.randint(len(self.data)) if self.is_train else call_index - 1
        left_call = torch.cat((self.data[0], self.data[left_call_index]))
        
        right_call_index = np.random.randint(len(self.data)) if self.is_train else call_index + 1
        right_call = torch.cat((self.data[right_call_index], self.data[-1]))

        data = torch.cat((left_call, call, right_call))
        x = data[len(left_call) + target_index - self.left_offset : len(left_call) + target_index + self.right_offset + 1]

        return x, y, ignored

