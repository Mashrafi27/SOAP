from aggregate_soap_generating_script import aggregate
import torch
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch.nn as nn

class MOFPropertyPredictor(nn.Module):
    """
    Simple neural network that takes SetTransformer output and predicts gas uptake.
    """
    
    def __init__(self, aggregated_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(aggregated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # Single output for gas uptake
        )
    
    def forward(self, aggregated_features):
        return self.predictor(aggregated_features).squeeze()

# Load the checkpoint
checkpoint = torch.load('optimized_models.pth', map_location='cpu')

# Initialize SetTransformer with your exact parameters
encoder_decoder = SetTransformerAggregation(
    channels=484,
    num_seed_points=1,
    num_encoder_blocks=1,
    num_decoder_blocks=1,
    heads=4,
    concat=True,
    layer_norm=True,
    dropout=0.3
)

# Load only the encoder_decoder weights
encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
encoder_decoder.eval()

property_predictor = MOFPropertyPredictor(484)
property_predictor.load_state_dict(checkpoint['property_predictor'])
property_predictor.eval()


soap_df = aggregate(encoder_decoder, property_predictor)



soap_df.to_csv('aggregate_soap_mofs.csv', index=False)  # `index=False` to avoid writing row numbers