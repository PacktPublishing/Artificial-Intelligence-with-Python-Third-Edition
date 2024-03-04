import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_length, embed_size)
        for pos in range(max_length):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                self.encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, :x.size(1), :]