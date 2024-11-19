import torch.nn as nn
# 28*28 ==> 9 ==> 28*28
class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        # N,784 with N is batch_size
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3))

        # N,3
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
