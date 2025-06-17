import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedAttentionMIL(nn.Module):
    """Enhanced Attention-based MIL pooling model with improved architecture."""
    def __init__(self, input_dim, embed_dim, attention_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Multi-layer embedding with residual connections
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention mechanism
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # Multi-head gated attention
        self.attention_V = nn.Linear(embed_dim, attention_dim)
        self.attention_U = nn.Linear(embed_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, num_heads)
        
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Enhanced classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * (num_heads + 2), embed_dim * 2),  # +2 for avg and max pooling
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch norm in embedding
        x_reshaped = x.view(-1, x.size(-1))
        H = self.embed(x_reshaped)
        H = H.view(batch_size, seq_len, -1)
        
        # Feature enhancement
        H_enhanced = self.feature_enhance(H.view(-1, H.size(-1))).view(batch_size, seq_len, -1)
        H = H + H_enhanced  # Residual connection
        
        # Multi-head gated attention
        A_V = torch.tanh(self.attention_V(H))  # (batch_size, seq_len, attention_dim)
        A_U = torch.sigmoid(self.attention_U(H))  # (batch_size, seq_len, attention_dim)
        A = A_V * A_U  # (batch_size, seq_len, attention_dim)
        
        # Multi-head attention weights
        A = self.attention_weights(A)  # (batch_size, seq_len, num_heads)
        
        # Apply mask to each head
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.num_heads)
        A = A.masked_fill(mask_expanded == 0, float('-inf'))
        A = torch.softmax(A, dim=1)  # (batch_size, seq_len, num_heads)
        
        # Multi-head pooling
        pooled_features = []
        for head in range(self.num_heads):
            head_attention = A[:, :, head].unsqueeze(-1)  # (batch_size, seq_len, 1)
            head_pooled = torch.sum(H * head_attention, dim=1)  # (batch_size, embed_dim)
            pooled_features.append(head_pooled)
        
        # Additional pooling strategies
        # Mask out padded positions for pooling
        H_masked = H * mask.unsqueeze(-1)
        
        # Average pooling (excluding padded positions)
        lengths = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        avg_pooled = H_masked.sum(dim=1) / lengths  # (batch_size, embed_dim)
        
        # Max pooling (set padded positions to very negative values)
        H_for_max = H_masked + (1 - mask.unsqueeze(-1)) * (-1e9)
        max_pooled = H_for_max.max(dim=1)[0]  # (batch_size, embed_dim)
        
        # Combine all pooled features
        all_features = pooled_features + [avg_pooled, max_pooled]
        Z = torch.cat(all_features, dim=1)  # (batch_size, embed_dim * (num_heads + 2))
        
        # Final prediction
        y_hat = self.classifier(Z).squeeze(-1)
        
        # Return average attention for interpretability
        avg_attention = A.mean(dim=-1)  # (batch_size, seq_len)
        
        return y_hat, avg_attention


class ResidualAttentionMIL(nn.Module):
    """AttentionMIL with residual connections and better regularization."""
    def __init__(self, input_dim, embed_dim, attention_dim, dropout=0.15):
        super().__init__()
        
        # Deeper embedding with residual connections
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        self.embed_blocks = nn.ModuleList([
            self._make_residual_block(embed_dim, dropout) for _ in range(3)
        ])
        
        # Improved attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, 1)
        )
        
        # Improved classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _make_residual_block(self, embed_dim, dropout):
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x, mask):
        # Initial projection
        H = self.input_projection(x)
        H = self.dropout(H)
        
        # Residual blocks
        for block in self.embed_blocks:
            residual = H
            H = block(H) + residual  # Residual connection
            H = F.relu(H)
        
        # Attention mechanism
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = A_V * A_U
        A = self.attention_weights(A).squeeze(-1)
        
        # Apply mask and softmax
        A = A.masked_fill(mask == 0, float('-inf'))
        A = torch.softmax(A, dim=1)
        
        # Weighted pooling
        Z = torch.sum(H * A.unsqueeze(-1), dim=1)
        
        # Classification
        y_hat = self.classifier(Z).squeeze(-1)
        
        return y_hat, A