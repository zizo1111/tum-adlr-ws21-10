import torch


def RMSE(estimate, gt):
    return torch.sqrt(torch.mean((estimate - gt) ** 2))


# TODO nll
def NLL(estimate, gt):
    pass
