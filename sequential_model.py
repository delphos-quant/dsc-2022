import numpy as np
from torch import nn, optim
import torch


def get_vals():
    x = np.array([float(x) for x in range(11)])
    y = np.array([1.6 * x + 4 + np.random.normal(10, 1) for x in X])
    return x, y


def get_model():
    seq_model = nn.Sequential(
        nn.Linear(1, 13),
        nn.Tanh(),
        nn.Linear(13, 1))
    return seq_model


def training_loop(n_epochs, optimiser, model, loss_fn, x_train, x_val, y_train, y_val):
    x_train, x_val, y_train, y_val = list(map(torch.tensor, [x_train, x_val, y_train, y_val]))

    for epoch in range(1, n_epochs + 1):
        output_train = model(x_train)  # forwards pass
        loss_train = loss_fn(output_train, y_train)  # calculate loss
        output_val = model(x_val)
        loss_val = loss_fn(output_val, y_val)

        optimiser.zero_grad()  # set gradients to zero
        loss_train.backward()  # backwards pass
        optimiser.step()  # update model parameters
        if epoch == 1 or epoch % 10000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")


def main(seq_model=None):
    x, y = get_vals()
    x, y = x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)
    x_train, y_train, x_val, y_val = x[:9], y[:9], x[9:], y[9:]

    if seq_model is None:
        seq_model = get_model()

    optimiser = optim.SGD(seq_model.parameters(), lr=1e-3)
    training_loop(
        n_epochs=5,
        optimiser=optimiser,
        model=seq_model,
        loss_fn=nn.MSELoss(),
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val)

    return seq_model
