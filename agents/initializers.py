import torch.nn as nn

def xavier_init(model):
    if type(model) == nn.Linear:
        nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0)
