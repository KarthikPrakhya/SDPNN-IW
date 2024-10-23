import torch


def regularized_l2_loss(yhat, y, model, beta):
    """
    The regularized_l2_loss function implements a regularized L2 loss function for NN training.

    @type yhat: torch.Tensor
    @param yhat: the predicted labels
    @type y: torch.Tensor
    @param y: the ground truth labels
    @type: torch.nn.Module
    @param model: the PyTorch model to use for computing the loss
    @type beta: float
    @param beta: the regularization parameter
    @rtype: torch.Tensor
    @return: the loss value
    """
    loss = 0.5 * torch.norm(yhat - y) ** 2

    # l2 norm on first layer weights, l1 squared norm on second layer
    for layer, p in enumerate(model.parameters()):
        loss += beta / 2 * torch.norm(p) ** 2

    return loss
