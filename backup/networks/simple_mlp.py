import torch.nn as nn

__all__ = ["simple_mlp"]


class Simple_MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, block, layers, num_classes=11):
        super().__init__()
        print(num_classes)
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(128, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = "fc"

    def forward(self, x):
        """Forward pass"""
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.fc(x)
        return x


def simple_mlp(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = Simple_MLP(None, None, **kwargs)
    return model
