import torch
import torch.nn as nn
from position_encoding import PositionalEncoding
from attention import MultiHeadAttention
import math
class Feedforward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)

        # Add skip connection, followed by LayerNorm
        out = self.layer_norm(out + x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = Feedforward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        out = self.dropout(self.feed_forward(attention))
        return out

    # Define the Transformer model by stacking Transformer blocks


class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, dropout_rate, max_length):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden_size, dropout_rate) for _ in range(num_layers)]
        )
        self.positional_encoding = PositionalEncoding(embed_size, max_length)

    def forward(self, x, mask):
        out = self.positional_encoding(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out