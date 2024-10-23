import torch


def one_hot(labels, num_classes=10):
    """
    The one_hot function takes the labels and one-hot encodes them.

    @type: Torch.Tensor
    @param labels: the labels to one-hot encode.
    @type: int
    @param num_classes: the number of classes for the labels
    @rtype: Torch.Tensor
    @return: the one-hot encoding of the labels
    """
    y = torch.eye(num_classes)
    return y[labels.long()]


def one_hot_signed(labels, num_classes=10):
    """
    The one_hot function takes the labels and one-hot encodes them in a signed fashion (+1, -1).

    @type: Torch.Tensor
    @param labels: the labels to one-hot encode in a signed fashion.
    @type: int
    @param num_classes: the number of classes for the labels
    @rtype: Torch.Tensor
    @return: the signed one-hot encoding of the labels
    """
    y = torch.eye(num_classes)
    return 2 * y[labels.long()] - 1


def identity(labels, num_classes=10):
    """
    The identity function takes the labels and returns them as it is. Placeholder if no encoding is desired.

    @type: Torch.Tensor
    @param labels: the labels
    @type: int
    @param num_classes: the number of classes for the labels
    @rtype: Torch.Tensor
    @return: the labels as they are
    """
    return labels