from torch import nn


class FCNetwork(nn.Module):
    """
    The FCNetwork class implements a two-layer fully-connected ReLU neural network (one hidden layer)
    """

    def __init__(self, num_neurons_hidden, num_classes=10, input_dim=784):
        self.num_classes = num_classes
        super(FCNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, num_neurons_hidden, bias=False), nn.ReLU())
        self.layer2 = nn.Linear(num_neurons_hidden, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer2(self.layer1(x))
        return out



