import torch
import torch.nn as nn


class PhishHookNet(nn.Module):
    """
    A simple feedforward neural network for phishing URL detection.

    The network has 3 hidden layers with ReLU activation functions and a final output layer with a sigmoid activation function.

    Args:
        input_size (int): The size of the input features.

    Returns:
        torch.Tensor: The output tensor with the predicted probability of the URL being a phishing URL.
    """

    def __init__(self, input_size):
        super(PhishHookNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x
