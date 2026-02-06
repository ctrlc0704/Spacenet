# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim=64):
        super().__init__()
        self.linear = nn.Linear(num_features, embed_dim)

    def forward(self, x):
        return self.linear(x).unsqueeze(1)  # [B, 1, D]


class SpaceNetTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        return self.encoder(x)


class SpaceNet(nn.Module):
    def __init__(self, num_features, num_classes=4):
        super().__init__()
        self.embedding = TabularEmbedding(num_features)
        self.transformer = SpaceNetTransformer()
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)[:, 0, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
