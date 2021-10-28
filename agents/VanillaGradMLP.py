import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.initializers import xavier_init

class VanillaGradMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5,
                 lr=0.05):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

    def init_weights(self):
        self.apply(xavier_init)


