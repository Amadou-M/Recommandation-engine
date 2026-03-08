"""
Définition partagée du modèle PyTorch
Utilisé par 04_tf_embeddings.py et l'API
"""
import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        u   = self.user_embedding(user)
        i   = self.item_embedding(item)
        dot = (u * i).sum(dim=1, keepdim=True)
        x   = torch.cat([u, i, dot], dim=1)
        return self.fc(x).squeeze()