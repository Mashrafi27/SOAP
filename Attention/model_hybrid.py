import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle

class AttentionFeatureExtractor(nn.Module):
    """Attention-based feature extractor for LightGBM."""
    def __init__(self, input_dim, embed_dim, attention_dim, feature_dim=128):
        super().__init__()
        
        # Embedding layers
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Attention mechanism
        self.attention_V = nn.Linear(embed_dim, attention_dim)
        self.attention_U = nn.Linear(embed_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch norm
        x_reshaped = x.view(-1, x.size(-1))
        H = self.embed(x_reshaped)
        H = H.view(batch_size, seq_len, -1)
        
        # Attention mechanism
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = A_V * A_U
        A = self.attention_weights(A).squeeze(-1)
        
        # Apply mask
        A = A.masked_fill(mask == 0, float('-inf'))
        A = torch.softmax(A, dim=1)
        
        # Attention-weighted pooling
        Z_attention = torch.sum(H * A.unsqueeze(-1), dim=1)
        
        # Additional statistical features
        # Mask out padded positions
        H_masked = H * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).float()
        
        # Mean pooling (excluding padding)
        Z_mean = H_masked.sum(dim=1) / lengths
        
        # Max pooling (excluding padding)
        H_for_max = H_masked + (1 - mask.unsqueeze(-1)) * (-1e9)
        Z_max = H_for_max.max(dim=1)[0]
        
        # Standard deviation (approximate)
        Z_std = torch.sqrt(((H_masked - Z_mean.unsqueeze(1)) ** 2).sum(dim=1) / lengths)
        
        # Extract features for each representation
        feat_attention = self.feature_extractor(Z_attention)
        feat_mean = self.feature_extractor(Z_mean)
        feat_max = self.feature_extractor(Z_max)
        feat_std = self.feature_extractor(Z_std)
        
        # Attention statistics
        attn_stats = torch.stack([
            A.max(dim=1)[0],  # Max attention
            A.mean(dim=1),    # Mean attention
            A.std(dim=1),     # Std attention
            (A > 0.1).sum(dim=1).float() / lengths.squeeze()  # Fraction with high attention
        ], dim=1)
        
        # Combine all features
        features = torch.cat([
            feat_attention, feat_mean, feat_max, feat_std, attn_stats
        ], dim=1)
        
        return features, A


class HybridAttentionLGB:
    """Hybrid model combining AttentionMIL feature extraction with LightGBM regression."""
    
    def __init__(self, input_dim, embed_dim=256, attention_dim=128, feature_dim=128):
        self.feature_extractor = AttentionFeatureExtractor(
            input_dim, embed_dim, attention_dim, feature_dim
        )
        self.lgb_model = None
        self.is_trained = False
        
        # Calculate total feature dimension
        self.total_feature_dim = feature_dim * 4 + 4  # 4 feature types + 4 attention stats
        
    def extract_features(self, dataloader, device='cpu'):
        """Extract features using the attention mechanism."""
        self.feature_extractor.eval()
        all_features = []
        all_targets = []
        
        with torch.no_grad():
            for x, mask, y in dataloader:
                x, mask = x.to(device), mask.to(device)
                features, _ = self.feature_extractor(x, mask)
                
                all_features.append(features.cpu().numpy())
                all_targets.append(y.numpy())
        
        return np.vstack(all_features), np.concatenate(all_targets)
    
    def train(self, train_dataloader, val_dataloader, device='cpu', lgb_params=None):
        """Train the hybrid model."""
        print("Extracting features from training data...")
        train_features, train_targets = self.extract_features(train_dataloader, device)
        
        print("Extracting features from validation data...")
        val_features, val_targets = self.extract_features(val_dataloader, device)
        
        # Default LightGBM parameters
        if lgb_params is None:
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(train_features, label=train_targets)
        val_data = lgb.Dataset(val_features, label=val_targets, reference=train_data)
        
        print(f"Training LightGBM with {train_features.shape[1]} features...")
        
        # Train LightGBM
        self.lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.is_trained = True
        print("LightGBM training completed!")
        
        return self.lgb_model
    
    def predict(self, dataloader, device='cpu'):
        """Make predictions using the hybrid model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        features, targets = self.extract_features(dataloader, device)
        predictions = self.lgb_model.predict(features, num_iteration=self.lgb_model.best_iteration)
        
        return predictions, targets
    
    def save_model(self, filepath_prefix):
        """Save both components of the hybrid model."""
        # Save feature extractor
        torch.save(self.feature_extractor.state_dict(), f"{filepath_prefix}_feature_extractor.pth")
        
        # Save LightGBM model
        self.lgb_model.save_model(f"{filepath_prefix}_lgb_model.txt")
        
        print(f"Hybrid model saved with prefix: {filepath_prefix}")
    
    def load_model(self, filepath_prefix, input_dim, embed_dim=256, attention_dim=128, feature_dim=128):
        """Load both components of the hybrid model."""
        # Load feature extractor
        self.feature_extractor = AttentionFeatureExtractor(
            input_dim, embed_dim, attention_dim, feature_dim
        )
        self.feature_extractor.load_state_dict(torch.load(f"{filepath_prefix}_feature_extractor.pth"))
        
        # Load LightGBM model
        self.lgb_model = lgb.Booster(model_file=f"{filepath_prefix}_lgb_model.txt")
        self.is_trained = True
        
        print(f"Hybrid model loaded from: {filepath_prefix}")


# Training function for the hybrid model
def train_hybrid_model(train_dataloader, val_dataloader, input_dim=485, pretrain_epochs=50):
    """Train the hybrid AttentionMIL + LightGBM model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    embed_dim = 256
    attention_dim = 128
    feature_dim = 128
    
    # Initialize hybrid model
    hybrid_model = HybridAttentionLGB(
        input_dim=input_dim,
        embed_dim=embed_dim,
        attention_dim=attention_dim,
        feature_dim=feature_dim
    )
    
    # Move feature extractor to device
    hybrid_model.feature_extractor = hybrid_model.feature_extractor.to(device)
    
    print("=== Step 1: Pre-training Attention Feature Extractor ===")
    
    # Pre-train the attention mechanism
    pretrain_attention_extractor(
        hybrid_model.feature_extractor, 
        train_dataloader, 
        val_dataloader, 
        device, 
        epochs=pretrain_epochs,
        feature_dim=feature_dim  # Pass feature_dim explicitly
    )
    
    print("=== Step 2: Training LightGBM on Attention Features ===")
    
    # LightGBM parameters optimized for your problem
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': 0,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_data_in_leaf': 20,
        'random_state': 42,
        'force_row_wise': True
    }
    
    # Train LightGBM with pre-trained features
    lgb_model = hybrid_model.train(
        train_dataloader, 
        val_dataloader, 
        device=device,
        lgb_params=lgb_params
    )
    
# Long training function for extended training
def train_hybrid_model_long(train_dataloader, val_dataloader, input_dim=485, 
                           pretrain_epochs=200, lgb_rounds=5000):
    """Train the hybrid model for extended time."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    embed_dim = 256
    attention_dim = 128
    feature_dim = 128
    
    # Initialize hybrid model
    hybrid_model = HybridAttentionLGB(
        input_dim=input_dim,
        embed_dim=embed_dim,
        attention_dim=attention_dim,
        feature_dim=feature_dim
    )
    
    # Move feature extractor to device
    hybrid_model.feature_extractor = hybrid_model.feature_extractor.to(device)
    
    print(f"=== Step 1: Pre-training Attention for {pretrain_epochs} epochs ===")
    
    # Pre-train the attention mechanism for longer
    pretrain_attention_extractor(
        hybrid_model.feature_extractor, 
        train_dataloader, 
        val_dataloader, 
        device, 
        epochs=pretrain_epochs,
        feature_dim=feature_dim
    )
    
    print(f"=== Step 2: Training LightGBM for {lgb_rounds} rounds ===")
    
    # Extract features
    train_features, train_targets = hybrid_model.extract_features(train_dataloader, device)
    val_features, val_targets = hybrid_model.extract_features(val_dataloader, device)
    
    # LightGBM parameters for longer training
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.01,  # Reduced for longer training
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': 100,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_data_in_leaf': 20,
        'random_state': 42,
        'force_row_wise': True
    }
    
    # Create datasets
    train_data = lgb.Dataset(train_features, label=train_targets)
    val_data = lgb.Dataset(val_features, label=val_targets, reference=train_data)
    
    # Train LightGBM for longer
    hybrid_model.lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=lgb_rounds,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),  # More patience
            lgb.log_evaluation(period=100)
        ]
    )
    
    hybrid_model.is_trained = True
    print("Extended hybrid model training completed!")
    
    return hybrid_model


def pretrain_attention_extractor(feature_extractor, train_dataloader, val_dataloader, device, epochs=50, feature_dim=128):
    """Pre-train the attention feature extractor using a simple regression head."""
    
    # Calculate total feature dimension: 4 feature types * feature_dim + 4 attention stats
    total_feature_dim = feature_dim * 4 + 4
    
    # Add a temporary regression head for pre-training
    temp_regressor = nn.Sequential(
        nn.Linear(total_feature_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)
    ).to(device)
    
    # Optimizer for both feature extractor and temp regressor
    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(temp_regressor.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        feature_extractor.train()
        temp_regressor.train()
        
        train_losses = []
        for x, mask, y in train_dataloader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            features, _ = feature_extractor(x, mask)
            
            # Predict using temp regressor
            predictions = temp_regressor(features).squeeze(-1)
            
            # Calculate loss
            loss = loss_fn(predictions, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(temp_regressor.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        feature_extractor.eval()
        temp_regressor.eval()
        
        val_losses = []
        with torch.no_grad():
            for x, mask, y in val_dataloader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                
                features, _ = feature_extractor(x, mask)
                predictions = temp_regressor(features).squeeze(-1)
                loss = loss_fn(predictions, y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print(f"Pre-training completed! Best Val Loss: {best_val_loss:.4f}")
    
    # Remove the temporary regressor (we only needed it for training)
    del temp_regressor


# Evaluation function for the hybrid model
def evaluate_hybrid_model(hybrid_model, test_dataloader):
    """Evaluate the hybrid model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictions, targets = hybrid_model.predict(test_dataloader, device)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    print(f"=== Hybrid Model Test Results ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


# Usage example:
"""
# Train hybrid model
hybrid_model = train_hybrid_model(train_dataloader, val_dataloader)

# Evaluate on test set
results = evaluate_hybrid_model(hybrid_model, test_dataloader)

# Save model
hybrid_model.save_model("hybrid_attention_lgb")
"""