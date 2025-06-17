import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    """Attention-based MIL pooling model."""
    def __init__(self, input_dim, embed_dim, attention_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attention_V = nn.Linear(embed_dim, attention_dim)
        self.attention_U = nn.Linear(embed_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, mask):
        H = self.embed(x)
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = A_V * A_U
        A = self.attention_weights(A).squeeze(-1)

        A = A.masked_fill(mask == 0, float('-inf'))
        A = torch.softmax(A, dim=1)
        Z = torch.sum(H * A.unsqueeze(-1), dim=1)

        y_hat = self.classifier(Z).squeeze(-1)
        return y_hat, A
