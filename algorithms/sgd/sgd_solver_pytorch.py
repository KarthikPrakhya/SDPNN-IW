import torch
import time
import numpy as np
from models.FCNetwork import FCNetwork
from utils.data.PrepareData import PrepareData
from torch.autograd import Variable
from loss_functions.regularized_l2_loss import regularized_l2_loss


def sgd_solver_pytorch(X, Y, m, beta, num_epochs, batch_size, learning_rate, loss_function, device):
    """
    The sgd_solver_pytorch function trains a two-layer fully-connected ReLU neural network (one hidden layer) using SGD.

    Author(s): Arda Sahiner et al.

    @type X: numpy.ndarray
    @param X: the data matrix that is of size n x d
    @type Y: numpy.ndarray
    @param Y: the labels matrix that is of size n x c
    @type m: int
    @param m: the number of neurons in the hidden layer
    @type beta: float
    @param beta: the regularization parameter for NN training
    @type num_epochs: int
    @param num_epochs: the number of epochs to run SGD for
    @type batch_size: int
    @param batch_size: the batch size to use with SGD
    @type learning_rate: float
    @param learning_rate: the initial learning rate (this is decremented as per a scheduler)
    @type loss_function: str
    @param loss_function: the loss function to use ('L2' or 'Cross_Entropy')
    @param device: the device to use for training (can be "cpu" or "cuda:x" where x is the index of GPU device)
    @return:
    """
    # Check the arguments
    n = X.shape[0]
    d = X.shape[1]
    c = Y.shape[1]

    if Y.shape[0] != n:
        raise ValueError('X and Y should have the same number of instances (n).')

    # create the model
    model = FCNetwork(m, c, d).to(device)

    # Create a train dataloader
    data = PrepareData(X, Y)
    train_dataloader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False,
        pin_memory=True, sampler=None)

    # Create a SGD optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, eps=1e-12, patience=100)

    # array for saving the loss
    losses = np.zeros((int(num_epochs * np.ceil(n / batch_size))))

    start = time.time()
    iter_no = 0
    for i in range(num_epochs):
        for ix, (_x, _y) in enumerate(train_dataloader):
            # =========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)

            # ========forward pass=====================================
            yhat = model(_x).float()

            if loss_function == 'L2':
                loss = regularized_l2_loss(yhat, _y, model, beta)
            else:
                raise ValueError('Loss function is not supported. It must be L2.')

            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            losses[iter_no] = loss.item()  # loss on the minibatch
            iter_no += 1

        if i % 10 == 0:
            scheduler.step(losses[iter_no - 1])
            print("Epoch [{}/{}], loss: {}".format(i, num_epochs, losses[iter_no - 1]))
    end = time.time()
    train_time = end - start
    return losses, train_time, model
