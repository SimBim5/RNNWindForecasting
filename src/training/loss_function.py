import torch
import torch.nn as nn

def mean_squared_error():
    return nn.MSELoss()

def mean_absolute_error():
    return nn.L1Loss()

def cross_entropy():
    return nn.CrossEntropyLoss()

def huber_loss():
    delta=1.0
    return nn.SmoothL1Loss(reduction='mean', beta=delta)

def symmetric_mean_absolute_percentage_error():
    def loss_function(predictions, targets):
        absolute_percentage_error = torch.abs((targets - predictions) / torch.max(torch.abs(targets), torch.abs(predictions)))
        return 200 * torch.mean(absolute_percentage_error)
    return loss_function

class GaussianNLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLoss, self).__init__()

    def forward(self, mean_predictions, variance_predictions, ground_truth):
        variance_predictions = torch.clamp(variance_predictions, min=1e-6)
        nll_loss = 0.5 * torch.log(2 * torch.pi * variance_predictions) + \
                   ((ground_truth - mean_predictions) ** 2) / (2 * variance_predictions)
        return torch.mean(nll_loss)

# Now, define a function to instantiate this class
def gaussian_negative_log_likelihood():
    return GaussianNLLoss()