from torch import nn
import torch
import numpy as np

class BirdCrossEntropyLoss(nn.Module):
    def __init__(self, device):
        super(BirdCrossEntropyLoss, self).__init__()
        self.rev_matrix = torch.tensor(np.array(
            [[0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
             [-0.25,  1., -0.3, -0.1, -0.1, -0.1, -0.1],
                [-0.02, -0.1,  1., -0.1, -0.1, -0.1, -0.1],
                [-0.25, -0.1, -0.3,  1., -0.1, -0.1, -0.1],
                [-0.25, -0.1, -0.3, -0.1,  1., -0.1, -0.1],
                [-0.25, -0.1, -0.3, -0.1, -0.1,  1., -0.1],
                [-0.25, -0.1, -0.3, -0.1, -0.1, -0.1,  1.]])).to(device)
        self.cet = nn.CrossEntropyLoss(reduction='none').to(device)
        self.device = device

    def forward(self, input, target, ignored): #include agreement somehow?? (results are not promising)
        ce_loss = self.cet(input, target)

        input_labels = torch.argmax(input, dim=1)
        target_labels = torch.argmax(target, dim=1)
        wanted = self.rev_matrix[target_labels, target_labels]
        actual = self.rev_matrix[target_labels, input_labels]
        cost = wanted - actual
        custom_loss = (1 + cost) * ce_loss * (~ignored)

        return custom_loss.mean()