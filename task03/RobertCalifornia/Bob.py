import numpy as np
from torch import nn
import torch

class Bob(nn.Module):
    def __init__(self, left_offset, right_offset):
        super(Bob, self).__init__()

        self.left_offset = left_offset
        self.right_offset = right_offset

        with open(f'../../dataset/original/feature_names.txt', 'r') as f:
            self.feature_names = np.array(f.read().splitlines())

        out_size = self.make_image_layers()
        out_size += self.make_one_d_layers()
        self.make_linear_layers(out_size)

    def make_image_layers(self):
        self.melspect_indices = torch.tensor(np.stack(
            (np.where(np.char.startswith(self.feature_names, 'raw_melspect_mean_'))[0],
             np.where(np.char.startswith(
                 self.feature_names, 'raw_melspect_std_'))[0],
             np.where(np.char.startswith(
                 self.feature_names, 'raw_melspect_mean_'))[0],
             np.where(np.char.startswith(self.feature_names, 'raw_melspect_std_'))[0])))

        self.contrast_indices = torch.tensor(np.stack(
            (np.where(np.char.startswith(self.feature_names, 'raw_contrast_mean_'))[0],
             np.where(np.char.startswith(
                 self.feature_names, 'raw_contrast_std_'))[0],
             np.where(np.char.startswith(
                 self.feature_names, 'raw_contrast_mean_'))[0],
             np.where(np.char.startswith(
                 self.feature_names, 'raw_contrast_std_'))[0])))

        indices = []
        for name in ['mfcc_', 'mfcc_d_', 'mfcc_d2_']:
            for type in ['raw_', 'cln_']:
                indices.append(np.where(np.char.startswith(
                    self.feature_names, f'{type}{name}mean'))[0])
                indices.append(np.where(np.char.startswith(
                    self.feature_names, f'{type}{name}std'))[0])
        self.mfcc_indices = torch.tensor(np.stack(indices))

        self.image_convs = nn.Sequential(
            ConvGroup(
                self.melspect_indices.shape[0] + self.contrast_indices.shape[0] + self.mfcc_indices.shape[0], 32),
            ConvGroup(32, 32),
            ConvGroup(32, 32),
            nn.Flatten()
        )
        out_size = self.image_convs(torch.tensor(np.zeros(
            (1, self.melspect_indices.shape[0] + self.contrast_indices.shape[0] + self.mfcc_indices.shape[0], self.left_offset+self.right_offset+1, self.melspect_indices.shape[1]), dtype=np.float32))).shape[1]
        return out_size

    def make_one_d_layers(self):
        indices = [
            *torch.tensor(np.where(np.char.startswith(self.feature_names, f'yin_'))[0]).split(1)]

        indices.append(np.where(np.char.startswith(
            self.feature_names, 'zcr_mean'))[0])
        indices.append(np.where(np.char.startswith(
            self.feature_names, 'zcr_std'))[0])

        for name in ['flatness_', 'centroid_', 'flux_', 'energy_', 'power_', 'bandwidth_']:
            for type in ['raw_', 'cln_']:
                indices.append(np.where(np.char.startswith(
                    self.feature_names, f'{type}{name}mean'))[0])
                indices.append(np.where(np.char.startswith(
                    self.feature_names, f'{type}{name}std'))[0])

        self.one_d_indices = torch.tensor(np.stack(indices))

        self.one_d_convs = nn.Sequential(
            ConvGroup(len(indices), 32, one_d=True),
            ConvGroup(32, 32, one_d=True),
            ConvGroup(32, 32, one_d=True, batch_norm=False),
            nn.Flatten()
        )
        out_size = self.one_d_convs(torch.tensor(np.zeros(
            (1, len(indices), self.left_offset+self.right_offset+1), dtype=np.float32))).shape[1]
        return out_size

    def make_linear_layers(self, conv_out_size):
        self.linear = nn.Sequential(
            nn.Linear(conv_out_size, 7),
        )

    def forward(self, x):
        convolved = []
        # convolved.append(x[:, :, self.left_offset+1, :].squeeze(1).squeeze(1))

        input_images, input_one_d = self.get_images(x)
        conv_image = self.image_convs(input_images)
        convolved.append(conv_image)
        conv_one_d = self.one_d_convs(input_one_d)
        convolved.append(conv_one_d)

        merged = torch.cat(convolved, dim=1)
        result = self.linear(merged)
        return result

    def get_images(self, x):
        input_melspect = x[:, :, self.melspect_indices]
        input_melspect = torch.swapdims(input_melspect, 1, 2).squeeze(2)

        input_contrast = x[:, :, self.contrast_indices]
        input_contrast = torch.swapdims(input_contrast, 1, 2).squeeze(2)
        input_contrast = input_contrast.repeat_interleave(9, dim=3)
        input_contrast = input_contrast[:, :, :, 1:-2]

        input_mfcc = x[:, :, self.mfcc_indices]
        input_mfcc = torch.swapdims(input_mfcc, 1, 2).squeeze(2)
        input_mfcc = input_mfcc.repeat_interleave(3, dim=3)

        input_images = torch.cat(
            (input_melspect, input_contrast, input_mfcc), dim=1)

        input_one_d = x[:, :, self.one_d_indices]
        input_one_d = torch.swapdims(input_one_d, 1, 2).squeeze(3)

        return (input_images, input_one_d)

    def to(self, device):
        self.melspect_indices.to(device)
        self.contrast_indices.to(device)
        self.mfcc_indices.to(device)
        self.one_d_indices.to(device)
        super(Bob, self).to(device)
        return self


class ConvGroup(nn.Module):
    def __init__(self, inf, outf, one_d=False, batch_norm=True, dropout=True):
        super(ConvGroup, self).__init__()

        if one_d:
            self.process = nn.Conv1d(
                inf, outf, kernel_size=3, stride=1, padding='same')
            self.scale = nn.MaxPool1d(kernel_size=2, stride=2)
            self.norm = nn.BatchNorm1d(outf) if batch_norm else nn.Identity()
            self.droput = nn.Dropout1d(0.3) if dropout else nn.Identity()
        else:
            self.process = nn.Conv2d(
                inf, outf, kernel_size=3, stride=1, padding='same')
            self.scale = nn.MaxPool2d(kernel_size=2, stride=2)
            self.norm = nn.BatchNorm2d(outf) if batch_norm else nn.Identity()
            self.droput = nn.Dropout2d(0.3) if dropout else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.process(x)
        x = self.scale(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.droput(x)
        return x


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
left_offset = 7
right_offset = 7
name = 'robert'
bob_dummy = Bob(left_offset, right_offset)
model_parameters = filter(lambda p: p.requires_grad, bob_dummy.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(bob_dummy, params)