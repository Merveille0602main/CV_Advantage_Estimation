"""Neural network construction shared by value and action predictors."""

DEFAULT_HIDDEN = (256, 256)


def scalar_mlp(input_dim, hidden=DEFAULT_HIDDEN):
    import torch
    layers, width = [], int(input_dim)
    for units in hidden:
        units = int(units)
        if units < 1:
            raise ValueError("hidden units must be positive")
        layers += [torch.nn.Linear(width, units), torch.nn.ReLU()]
        width = units
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)
