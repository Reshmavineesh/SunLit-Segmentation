import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
from torchvision.models.segmentation import deeplabv3_resnet101

# Model
model = deeplabv3_resnet101()

# Loss functions
cross_entropy = nn.CrossEntropyLoss()
bce = nn.BCELoss()


# Optimizer
def optimizer(optimizer, lr, weight_decay):
    if optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    if optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Accuracy
def accuracy(outputs, masks):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == masks).sum().item()
    total = masks.size(0) * masks.size(1) * masks.size(2)
    return correct / total
