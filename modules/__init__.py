import lightning as L
import torch.nn as nn


class Identity(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)
