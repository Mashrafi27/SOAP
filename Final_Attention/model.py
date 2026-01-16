# training_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.nn.aggr import SetTransformerAggregation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

import wandb

from models import SetTransformer


# Import your existing functions
from aggregate_soap_generating_script import aggregate
from soap import read_cif, S, set_transform_aggregation

# from soft_tree import SoftTreeEnsemble   # <-- new

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MOFDataManager:
    """
    Manages train/validation/test split for MOF datasets and handles data loading.
    """
    
    def __init__(self, 
                 folder_path: str = '../CIF_files',
                 target_csv: str = "../id_labels.csv",
                 val_size: float = 0.0,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the data manager with three-way split.
        
        Args:
            folder_path: Path to CIF files
            target_csv: Path to CSV with target properties
            val_size: Fraction for validation (from total)
            test_size: Fraction for testing (from total)
            random_state: Random state for reproducibility
        """
        self.folder_path = folder_path
        self.target_csv = target_csv
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Ensure splits are valid
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")
        
        # Get all CIF files
        all_cif_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.cif')
        ]
        
        # Load targets first
        self.targets = self._load_targets()
        
        # Filter CIF files to only include those with targets
        self.all_files = []
        missing_targets = []
        
        for file_path in all_cif_files:
            filename = os.path.basename(file_path)
            if filename in self.targets:
                self.all_files.append(file_path)
            else:
                missing_targets.append(filename)
        
        if missing_targets:
            print(f"Warning: {len(missing_targets)} CIF files don't have targets and will be excluded")
            if len(missing_targets) <= 10:
                print(f"Files without targets: {missing_targets}")
        
        print(f"Total CIF files found: {len(all_cif_files)}")
        print(f"CIF files with targets: {len(self.all_files)}")
        
        if len(self.all_files) == 0:
            raise ValueError("No CIF files found with matching targets!")
        
        # Three-way split: Train / Validation / Test
        # First split: separate test set
        self.train_files, self.test_files = train_test_split(
            self.all_files,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: separate validation from training
        # Adjust validation size relative to remaining data
        # val_size_adjusted = val_size / (1 - test_size)
        # self.train_files, self.val_files = train_test_split(
        #     train_val_files,
        #     test_size=val_size_adjusted,
        #     random_state=random_state
        # )
        
        print(f"Data split:")
        print(f"  Train files: {len(self.train_files)} ({len(self.train_files)/len(self.all_files)*100:.1f}%)")
        # print(f"  Validation files: {len(self.val_files)} ({len(self.val_files)/len(self.all_files)*100:.1f}%)")
        print(f"  Test files: {len(self.test_files)} ({len(self.test_files)/len(self.all_files)*100:.1f}%)")
        
        # Verify no overlap
        train_set = set(self.train_files)
        # val_loss = set(self.val_files) 
        test_set = set(self.test_files)
        
        # assert len(train_set & val_set) == 0, "Train and validation sets overlap!"
        assert len(train_set & test_set) == 0, "Train and test sets overlap!"
        # assert len(val_set & test_set) == 0, "Validation and test sets overlap!"
        
        print("✓ Verified no data leakage between splits")
    
    def _load_targets(self) -> Dict[str, float]:
        """Load target values from CSV."""
        if self.target_csv and os.path.exists(self.target_csv):
            print(f"Loading targets from: {self.target_csv}")
            df = pd.read_csv(self.target_csv)
            
            print(f"CSV columns: {df.columns.tolist()}")
            print(f"CSV shape: {df.shape}")
            print(f"First few rows:")
            print(df.head())
            
            # For your specific file format: id, label
            if 'id' in df.columns and 'label' in df.columns:
                filename_col = 'id'
                target_col = 'label'
                print(f"Using 'id' column for filenames and 'label' column for targets")
            else:
                # Fallback
                filename_cols = ['filename', 'file', 'name', 'id', 'cif_name', 'structure_id']
                target_cols = ['target', 'label', 'value', 'property', 'y', 'output']
                
                filename_col = None
                target_col = None
                
                for col in df.columns:
                    if col.lower() in filename_cols or 'file' in col.lower() or 'name' in col.lower():
                        filename_col = col
                        break
                
                for col in df.columns:
                    if col.lower() in target_cols or 'target' in col.lower() or 'label' in col.lower():
                        target_col = col
                        break
                
                if filename_col is None:
                    filename_col = df.columns[0]
                    print(f"Warning: Using first column '{filename_col}' as filename column")
                
                if target_col is None:
                    target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    print(f"Warning: Using column '{target_col}' as target column")
                
                print(f"Using filename column: '{filename_col}'")
                print(f"Using target column: '{target_col}'")
            
            # Create the mapping
            targets_dict = {}
            for _, row in df.iterrows():
                # Handle your specific format: strip trailing spaces and add .cif
                filename_raw = str(row[filename_col])
                filename_clean = filename_raw.strip()  # Remove trailing/leading spaces
                target_value = float(row[target_col])
                
                # Add .cif extension since your filenames don't have it
                if not filename_clean.endswith('.cif'):
                    filename_clean += '.cif'
                
                targets_dict[filename_clean] = target_value
            
            print(f"Loaded {len(targets_dict)} target values")
            print(f"Target range: {min(targets_dict.values()):.3f} to {max(targets_dict.values()):.3f}")
            
            # Show a few examples of the filename mapping
            print("Example filename mappings:")
            for i, (filename, target) in enumerate(list(targets_dict.items())[:3]):
                print(f"  '{filename}' -> {target}")
            
            return targets_dict
        else:
            print("ERROR: Target CSV file not found or doesn't exist!")
            print(f"Expected file: {self.target_csv}")
            raise FileNotFoundError(f"Target file {self.target_csv} not found")
    
    def get_train_files(self) -> List[str]:
        """Get training file paths."""
        return self.train_files
    
    # def get_val_files(self) -> List[str]:
    #     """Get validation file paths."""
    #     return self.val_files
    
    def get_test_files(self) -> List[str]:
        """Get test file paths."""
        return self.test_files
    
    def get_targets_for_files(self, file_paths: List[str]) -> List[float]:
        """Get target values for given file paths."""
        targets = []
        missing_targets = []
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            
            if filename in self.targets:
                targets.append(self.targets[filename])
            else:
                missing_targets.append(filename)
        
        if missing_targets:
            raise ValueError(f"Missing targets for files: {missing_targets}. "
                           "All files should have targets in the CSV.")
        
        return targets

class ModifiedAggregateFunction:
    """
    Modified version of your aggregate function that works with train/test splits.
    """
    
    def __init__(self, columns_module):
        """
        Initialize with your columns module.
        """
        self.columns_module = columns_module
    
    def aggregate_files(self, encoder_decoder, file_list: List[str], all_soap_columns: List[str]) -> pd.DataFrame:
        """Modified aggregate function with fixed column padding."""
        from tqdm import tqdm
        
        soap_df = pd.DataFrame()
        
        for file_path in tqdm(file_list, desc="Processing files"):
            try:
                filename, structure, species = read_cif(file_path)
                soap_out, soap = S(structure, species)
                
                if soap_out.size == 0:
                    continue
                
                # Create SOAP features with current columns
                current_columns = self.columns_module.slice_column(soap, list(species))
                
                # Create DataFrame with current features
                current_features = soap_out.mean(axis=0)  # Or however you want to aggregate before SetTransformer
                df_current = pd.DataFrame([current_features], columns=current_columns)
                
                # Pad with zeros for missing columns
                df_padded = pd.DataFrame(columns=all_soap_columns, index=[0])
                df_padded = df_padded.fillna(0)  # Fill with zeros
                
                # Update with actual values
                for col in current_columns:
                    if col in df_padded.columns:
                        df_padded[col] = df_current[col].iloc[0]
                
                # Now apply SetTransformer aggregation on the padded features
                padded_features = df_padded.values  # Shape: (1, 484)
                aggr_out = set_transform_aggregation(padded_features, encoder_decoder)
                
                # Create final DataFrame
                df = pd.DataFrame([aggr_out], columns=[f'agg_feature_{i}' for i in range(len(aggr_out))])
                df['filename'] = filename
                
                soap_df = pd.concat([soap_df, df], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return soap_df

class RFLossFunction:
    """
    Random Forest-based loss function for optimization.
    """
    
    def __init__(self, 
                 n_estimators: int = 50,
                 max_depth: int = 10,
                 random_state: int = 42,
                 loss_metric: str = 'r2'):
        """
        Initialize RF loss function.
        """
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'n_jobs': 1
        }
        self.loss_metric = loss_metric
        self.scaler = StandardScaler()
    
    def compute_loss(self, features: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute RF-based loss.
        """
        if len(features) < 4:  # Need minimum samples
            return 1.0
        
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.3, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train RF
            rf = RandomForestRegressor(**self.rf_params)
            rf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = rf.predict(X_val_scaled)
            
            # Compute loss
            if self.loss_metric == 'r2':
                r2 = r2_score(y_val, y_pred)
                return 1.0 - r2  # Convert to loss (lower is better)
            elif self.loss_metric == 'mae':
                return mean_absolute_error(y_val, y_pred)
            elif self.loss_metric == 'mse':
                return mean_squared_error(y_val, y_pred)
            else:
                return 1.0
                
        except Exception as e:
            print(f"Error computing RF loss: {e}")
            return 1.0

class MOFSetTransformerTrainer:
    """
    Main training class for optimizing SetTransformerAggregation parameters.
    """
    
    def __init__(self, 
             data_manager: MOFDataManager,
             soap_dim: int = None,
             aggregator_params: Optional[Dict] = None,
             loss_metric: str = 'mse'):  # Changed default to mse
        """
        Initialize the trainer.
        """
        self.data_manager = data_manager
        self.soap_dim = soap_dim
        self.loss_metric = loss_metric
        
        # Import your columns module
        try:
            import columns
            self.columns_module = columns
        except ImportError:
            print("Warning: Could not import columns module. Creating mock.")
            self.columns_module = self._create_mock_columns()
        
        # Initialize modified aggregate function
        self.aggregate_fn = ModifiedAggregateFunction(self.columns_module)
        
        # Default aggregator parameters
        default_params = {
            'channels': 484,  # Will be updated with soap_dim
            'num_seed_points': 4,
            'num_encoder_blocks': 2,
            'num_decoder_blocks': 2,
            'heads': 4,
            'concat': True,
            'layer_norm': True,
            'dropout': 0.1
        }
    
        if aggregator_params:
            default_params.update(aggregator_params)
        
        self.aggregator_params = default_params
        
        # Will be initialized during training
        self.encoder_decoder = None
        # self.property_predictor = None  # New prediction head
        self.criterion = nn.MSELoss()   # Replace RF loss with MSE
        # self.criterion = nn.L1Loss()
        # self.criterion = nn.SmoothL1Loss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_r2': [],
            'test_r2': [],
            'train_mae': [],
            'test_mae': []
        }
    def determine_all_soap_columns(self) -> List[str]:
        """Determine all possible SOAP columns across ALL MOFs in the dataset."""
        from columns import slice_column
        
        print("Determining all unique SOAP columns across ENTIRE dataset...")
        all_columns = set()
        max_atoms = 0
        
        # Get ALL files from all splits
        all_files = (self.data_manager.get_train_files() + 
                    # self.data_manager.get_val_files() + 
                    self.data_manager.get_test_files())
        
        print(f"Processing ALL {len(all_files)} files to find unique columns...")
        
        for i, file_path in enumerate(all_files):
            if i % 100 == 0:  # Progress every 100 files
                print(f"Processed {i}/{len(all_files)} files, found {len(all_columns)} unique columns so far...")
            
            try:
                filename, structure, species = read_cif(file_path)
                soap_out, soap = S(structure, species)

                num_atoms = soap_out.shape[0]
                if num_atoms > max_atoms:
                    max_atoms = num_atoms
                
                # Get columns for this specific MOF
                columns = slice_column(soap, list(species))
                all_columns.update(columns)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        all_columns_list = sorted(list(all_columns))
        print(f"Found {len(all_columns_list)} unique SOAP columns across ALL files")
        
        self.all_soap_columns = all_columns_list
        self.soap_dim = len(all_columns_list)
        self.max_atoms = max_atoms
        return all_columns_list

    def _create_mock_columns(self):
        """Create a mock columns module for testing."""
        class MockColumns:
            @staticmethod
            def slice_column(soap, species):
                n_features = soap.get_number_of_features()
                return [f'feature_{i}' for i in range(n_features)]
        
        return MockColumns()
    
    # def _determine_soap_dim(self) -> int:
    #     """Determine SOAP dimension by processing one sample."""
    #     print("Determining SOAP dimension...")
        
    #     sample_file = self.data_manager.get_train_files()[0]
    #     filename, structure, species = read_cif(sample_file)
    #     soap_out, soap = S(structure, species)
        
    #     soap_dim = soap_out.shape[1]
    #     print(f"Determined SOAP dimension: {soap_dim}")
        
    #     return soap_dim

    # def _determine_soap_dim(self) -> int:
    #     """Determine the maximum number of atoms across all MOFs."""
    #     print("Determining max number of atoms (SOAP rows)...")

    #     max_atoms = 0
    #     for file in self.data_manager.get_train_files():
    #         filename, structure, species = read_cif(file)
    #         soap_out, _ = S(structure, species)

    #         if soap_out.shape[0] > max_atoms:
    #             max_atoms = soap_out.shape[0]

    #     print(f"Max number of atoms across MOFs: {max_atoms}")
    #     return max_atoms
    
    def _initialize_encoder_decoder(self):
        """Initialize the SetTransformerAggregation model."""
        # if self.soap_dim is None:
        #     self.soap_dim = self._determine_soap_dim()
        
        # Update channels parameter
        # self.aggregator_params['channels'] = self.soap_dim
        self.aggregator_params['channels'] = len(self.all_soap_columns)
        
        # Initialize model
        # self.encoder_decoder = SetTransformerAggregation(**self.aggregator_params)
        
        
        self.encoder_decoder = SetTransformer(dim_input = 484, num_outputs = 1, dim_output = 1) #, num_inds=32, dim_hidden=256, num_heads=4)
        # self.encoder_decoder.load_state_dict(torch.load("optimized_models_fast60.pth"))

        print(f"Initialized SetTransformerAggregation with parameters:")
        for key, value in self.aggregator_params.items():
            print(f"  {key}: {value}")
    
    def _process_split_data(self, file_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a list of files and return features and targets.
        """
        soap_df = self.aggregate_fn.aggregate_files(self.encoder_decoder, file_list, self.all_soap_columns)

        if soap_df.empty:
            return np.array([]), np.array([])
        
        # Extract features (all columns except filename)
        feature_cols = [col for col in soap_df.columns if col != 'filename']
        features = soap_df[feature_cols].values
        
        # Get targets
        filenames = soap_df['filename'].tolist()
        file_paths = [os.path.join(self.data_manager.folder_path, fname) for fname in filenames]
        targets = np.array(self.data_manager.get_targets_for_files(file_paths))
        
        return features, targets
    
    def _compute_metrics(self, features: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """Compute R² and MAE metrics."""
        try:
            if len(features) < 4:
                return -1.0, 1e6
            
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.3, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train_scaled, y_train)
            
            y_pred = rf.predict(X_val_scaled)
            
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            return r2, mae
        except:
            return -1.0, 1e6
    
    # def _pad_soap_features(self, soap_out, soap, species):
    #     """Pad SOAP features to fixed dimension."""
    #     current_columns = self.columns_module.slice_column(soap, list(species))
        
    #     # Create padded feature vector
    #     padded_features = np.zeros(len(self.all_soap_columns))
        
    #     # Fill in actual values
    #     for i, col in enumerate(current_columns):
    #         if col in self.all_soap_columns:
    #             col_idx = self.all_soap_columns.index(col)
    #             padded_features[col_idx] = soap_out[:, i].mean()  # Average across atoms
        
    #     return padded_features

    def _pad_soap_features(self, soap_out, soap, species):
        """
        Return a padded 2D SOAP matrix of shape [n_atoms, 484],
        where missing columns are filled with zeros.
        """
        current_columns = self.columns_module.slice_column(soap, list(species))
        n_atoms = soap_out.shape[0]
        
        padded_features = np.zeros((self.max_atoms, len(self.all_soap_columns)))  # shape: [n_atoms, 484]

        for i, col in enumerate(current_columns):
            if col in self.all_soap_columns:
                col_idx = self.all_soap_columns.index(col)
                padded_features[:n_atoms, col_idx] = soap_out[:, i]  # fill only existing atoms

        return padded_features  # shape: [n_atoms, 484]

    def _initialize_models(self):
        """Initialize both SetTransformer and prediction head."""
        # Determine columns if not done
        if self.all_soap_columns is None:
            self.determine_all_soap_columns()
        
        # Initialize SetTransformer
        self.aggregator_params['channels'] = self.soap_dim
        
        # Make sure heads divides channels evenly
        if self.soap_dim % self.aggregator_params['heads'] != 0:
            for heads in [1, 2, 3, 4, 6, 8, 12, 16]:
                if self.soap_dim % heads == 0:
                    self.aggregator_params['heads'] = heads
                    break
        
        self.encoder_decoder = SetTransformer(dim_input = 484, num_outputs = 1, dim_output = 1).to(device) #, num_inds=32, dim_hidden=256, num_heads=4).to(device)
        self.encoder_decoder.load_state_dict(torch.load("optimized_models_fast100.pth")['encoder_decoder'])
        self.encoder_decoder.to(device)
        # Initialize prediction head
        aggregated_dim = self.aggregator_params['num_seed_points'] * self.aggregator_params['channels']
        if self.aggregator_params['concat']:
            # SetTransformer concatenates seed points when concat=True
            pass  # aggregated_dim is correct
        else:
            aggregated_dim = self.aggregator_params['channels']
        
        # self.property_predictor = MOFPropertyPredictor(aggregated_dim).to(device)
        # self.property_predictor = NODE(aggregated_dim).to(device)

        # self.property_predictor = SoftTreeEnsemble(
        #     in_dim      = aggregated_dim,
        #     n_trees     = 64,      # ↔ hyper-param
        #     depth       = 6        # ↔ hyper-param (2**6 = 64 leaves)
        # ).to(device)
        
        print(f"Initialized SetTransformerAggregation with:")
        print(f"  SOAP dimension: {self.soap_dim}")
        print(f"  Unique columns: {len(self.all_soap_columns)}")
        for key, value in self.aggregator_params.items():
            print(f"  {key}: {value}")
        print(f"Initialized prediction head with input dim: {aggregated_dim}")

    def forward_pass(self, file_list, max_files=None):
        """Forward pass through both SetTransformer and prediction head."""
        aggregated_outputs = []
        targets = []
        
        files_to_process = file_list[:max_files] if max_files else file_list
    
        for file_path in files_to_process:
            try:
                filename, structure, species = read_cif(file_path)
                soap_out, soap = S(structure, species)
                
                if soap_out.size == 0:
                    continue
                
                # Pad SOAP features to fixed dimension
                padded_soap = self._pad_soap_features(soap_out, soap, species)
                
                # SetTransformer aggregation
                soap_tensor = torch.FloatTensor(padded_soap).to(device)  
                soap_tensor = soap_tensor.unsqueeze(0)
                group_indices = torch.zeros(soap_tensor.size(0), dtype=torch.long).to(device)  # Single group
                prediction = self.encoder_decoder(soap_tensor)
                
                prediction = prediction.view(-1)  # shape: (1,)
                aggregated_outputs.append(prediction)
                
                # Get target
                target = self.data_manager.targets[os.path.basename(file_path)]
                targets.append(target)
                
            except Exception as e:
                print(f"Error in forward pass for {file_path}: {e}")
                continue
        
        # if not aggregated_features:
        #     return None, None
        
        # Stack features and targets
        # predictions = torch.FloatTensor(aggregated).to(device)
        # predictions = torch.tensor(aggregated, dtype=torch.float32, device=device)
        predictions = torch.cat(aggregated_outputs, dim=0)  # shape: (batch_size,)
        targets_tensor = torch.FloatTensor(targets).to(device)
        
        # Predict gas uptake
        # predictions = self.property_predictor(features_tensor)
        
        return predictions, targets_tensor
    
    def train(self, 
          epochs: int = 30,
          learning_rate: float = 0.001,
          weight_decay: float = 1e-5,
          patience: int = 10,
          batch_size: int = 50,  # Process files in batches
          verbose: bool = True) -> Dict:
        """
        Train end-to-end with simple MSE loss.
        """
        
        # Initialize models
        self._initialize_models()
        
        # Optimizer for both models
        all_params = list(self.encoder_decoder.parameters())# + list(self.property_predictor.parameters())
        optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        train_files = self.data_manager.get_train_files()
        test_files = self.data_manager.get_test_files()
        
        best_test_loss = float('inf')
        patience_counter = 0
        best_model_states = None
        
        print(f"Starting end-to-end training for {epochs} epochs...")
        print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
        print(f"Using batch size: {batch_size}")
        
        for epoch in range(epochs):
            # Training
            self.encoder_decoder.train()
            # self.property_predictor.train()
            
            train_predictions, train_targets = self.forward_pass(train_files)#, max_files=batch_size)
            if train_predictions is None:
                print(f"No training data processed in epoch {epoch+1}")
                continue
                
            train_loss = self.criterion(train_predictions, train_targets)
            
            # Backprop
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
        
            # Validation
            self.encoder_decoder.eval()
            # self.property_predictor.eval()
            
            with torch.no_grad():
                test_predictions, test_targets = self.forward_pass(test_files)#, max_files=batch_size//2)
                if test_predictions is not None:
                    test_loss = self.criterion(test_predictions, test_targets)
                    
                    # Calculate metrics
                    train_r2 = r2_score(train_targets.cpu().numpy(), train_predictions.cpu().numpy())
                    test_r2 = r2_score(test_targets.cpu().numpy(), test_predictions.cpu().numpy())
                    train_mae = mean_absolute_error(train_targets.cpu().numpy(), train_predictions.cpu().numpy())
                    test_mae = mean_absolute_error(test_targets.cpu().numpy(), test_predictions.cpu().numpy())
                else:
                    test_loss = torch.tensor(float('inf'))
                    train_r2 = test_r2 = -1
                    train_mae = test_mae = 1e6
            
            scheduler.step(train_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss.item())
            self.history['test_loss'].append(test_loss.item())
            self.history['train_r2'].append(train_r2)
            self.history['test_r2'].append(test_r2)
            self.history['train_mae'].append(train_mae)
            self.history['test_mae'].append(test_mae)

            # Add wandb logging right after:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss.item(),
                "test_loss": test_loss.item(),
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # Print progress
            if verbose and (epoch % 1 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Test Loss: {test_loss:.6f}, "
                    f"Train R²: {train_r2:.4f}, "
                    f"Test R²: {test_r2:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss.item()
                patience_counter = 0
                best_model_states = {
                    'encoder_decoder': self.encoder_decoder.state_dict().copy(),
                    # 'property_predictor': self.property_predictor.state_dict().copy()
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_states is not None:
                    self.encoder_decoder.load_state_dict(best_model_states['encoder_decoder'])
                    # self.property_predictor.load_state_dict(best_model_states['property_predictor'])
                break
            if epoch == 100:
                torch.save({
                    'encoder_decoder': self.encoder_decoder.state_dict(),
                    # 'property_predictor': results['property_predictor'].state_dict(),
                }, 'optimized_models_fast100.pth')
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Best test loss: {best_test_loss:.6f}")
        print(f"Final test R²: {test_r2:.4f}")
        print(f"Final test MAE: {test_mae:.4f}")
    
        results = {
            'encoder_decoder': self.encoder_decoder,
            # 'property_predictor': self.property_predictor,
            'history': self.history,
            'best_test_loss': best_test_loss,
            'final_metrics': {
                'train': {'r2': train_r2, 'mae': train_mae},
                'test': {'r2': test_r2, 'mae': test_mae}
            }
        }
        
        return results
    
    def plot_training_history(self):
        """Plot training history for end-to-end neural network training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.history['train_loss']:
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', alpha=0.8)
            axes[0, 0].plot(epochs, self.history['test_loss'], 'r-', label='Test Loss', alpha=0.8)
            axes[0, 0].set_title('Training and Test Loss (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # R² curves
        if self.history['train_r2']:
            axes[0, 1].plot(epochs, self.history['train_r2'], 'b-', label='Train R²', alpha=0.8)
            axes[0, 1].plot(epochs, self.history['test_r2'], 'r-', label='Test R²', alpha=0.8)
            axes[0, 1].set_title('R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
    
        # MAE curves
        if self.history['train_mae']:
            axes[1, 0].plot(epochs, self.history['train_mae'], 'b-', label='Train MAE', alpha=0.8)
            axes[1, 0].plot(epochs, self.history['test_mae'], 'r-', label='Test MAE', alpha=0.8)
            axes[1, 0].set_title('Mean Absolute Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter histogram
        if self.encoder_decoder is not None: # and self.property_predictor is not None:
            all_params = []
            # Get parameters from both models
            for param in self.encoder_decoder.parameters():
                all_params.extend(param.data.cpu().numpy().flatten())
            # for param in self.property_predictor.parameters():
            #     all_params.extend(param.data.cpu().numpy().flatten())
            
            axes[1, 1].hist(all_params, bins=50, alpha=0.7, color='purple')
            axes[1, 1].set_title('Final Parameter Distribution\n(Both Models)')
            axes[1, 1].set_xlabel('Parameter Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class MOFPropertyPredictor(nn.Module):
    """
    Simple neural network that takes SetTransformer output and predicts gas uptake.
    """
    
    def __init__(self, aggregated_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(aggregated_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # Single output for gas uptake
        )
    
    def forward(self, aggregated_features):
        return self.predictor(aggregated_features).squeeze()

import torch
import torch.nn as nn
import torch.nn.functional as F

class NODE(nn.Module):
    def __init__(self, input_dim, num_trees=128, depth=8, tree_output_dim=1):
        super(NODE, self).__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.tree_output_dim = tree_output_dim

        self.n_leaf = 2 ** depth  # Number of leaves per tree
        self.n_total_leaf = self.n_leaf * num_trees

        # Each internal node chooses a feature to split
        self.feature_selectors = nn.Parameter(torch.randn(num_trees, depth, input_dim))
        self.feature_bias = nn.Parameter(torch.zeros(num_trees, depth))

        # Leaf outputs
        self.leaf_responses = nn.Parameter(torch.zeros(num_trees, self.n_leaf, tree_output_dim))

    def forward(self, x):  # x: [batch_size, input_dim]
        batch_size = x.size(0)

        # Compute routing probabilities for each tree
        all_probs = []
        for t in range(self.num_trees):
            h = x  # shape: [B, input_dim]

            probs = []
            for d in range(self.depth):
                selector = self.feature_selectors[t, d]  # shape: [input_dim]
                bias = self.feature_bias[t, d]           # scalar

                # Project x using selector
                route = torch.matmul(h, selector) + bias  # shape: [B]
                prob = torch.sigmoid(route).unsqueeze(1)  # shape: [B, 1]
                probs.append(prob)

            probs = torch.cat(probs, dim=1)  # shape: [B, depth]

            # Build all leaf paths
            leaf_paths = self._get_leaf_paths().to(x.device)  # [n_leaf, depth]

            # Compute final probability of landing in each leaf
            leaf_probs = []
            for path in leaf_paths:  # over 2^depth leaves
                path_probs = probs.clone()
                mask = path.unsqueeze(0).expand_as(path_probs) == 0  # shape: [B, depth]
                path_probs = torch.where(mask, 1 - path_probs, path_probs)
                leaf_prob = path_probs.prod(dim=1)  # shape: [B]
                leaf_probs.append(leaf_prob.unsqueeze(1))

            leaf_probs = torch.cat(leaf_probs, dim=1)  # shape: [B, n_leaf]
            all_probs.append(leaf_probs.unsqueeze(1))  # shape: [B, 1, n_leaf]

        # Concatenate probs from all trees
        all_probs = torch.cat(all_probs, dim=1)  # [B, num_trees, n_leaf]

        # Compute outputs per leaf
        leaf_values = self.leaf_responses  # [num_trees, n_leaf, output_dim]
        out = torch.einsum('btn,tno->bto', all_probs, leaf_values)  # [B, num_trees, output_dim]

        return out.mean(dim=1)  # Mean over trees → [B, output_dim]

    def _get_leaf_paths(self):
        """ Return binary leaf paths. For depth=2: [[0,0],[0,1],[1,0],[1,1]] """
        leaf_paths = []
        for i in range(2 ** self.depth):
            path = [int(b) for b in bin(i)[2:].zfill(self.depth)]
            leaf_paths.append(path)
        return torch.tensor(leaf_paths, dtype=torch.float32)


# def determine_all_soap_columns(file_list: List[str]) -> List[str]:
#     """Determine all possible SOAP columns across all MOFs."""
#     print("Determining all unique SOAP columns across dataset...")
#     all_columns = set()
    
#     for file_path in tqdm(file_list[:100], desc="Sampling files for columns"):  # Sample first 100 files
#         try:
#             filename, structure, species = read_cif(file_path)
#             soap_out, soap = S(structure, species)
#             columns = slice_column(soap, list(species))  # Your existing function
#             all_columns.update(columns)
#         except Exception as e:
#             continue
    
#     all_columns_list = sorted(list(all_columns))
#     print(f"Found {len(all_columns_list)} unique SOAP columns")
#     return all_columns_list

def main():
    """Main function demonstrating the training pipeline."""
    print("=== MOF SetTransformer Training Pipeline ===\n")

    wandb.login(key="2f1002514629c6ffe0f9d6d1fe10914998145aa7")
    # Initialize wandb
    wandb.init(
        project="mof-settransformer",
        config={
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 100,
            "patience": 20,
            "num_seed_points": 1,
            "num_encoder_blocks": 2,
            "num_decoder_blocks": 1,
            "heads": 4,
            "dropout": 0.3,
            "loss_metric": "mae"
        }
    )
    
    
    # Initialize data manager with three-way split
    data_manager = MOFDataManager(
        folder_path='../CIF_files',  # Adjust path as needed
        target_csv="../id_labels.csv",  # Your target file
        val_size=0.0,   # 20% for validation
        test_size=0.2,  # 20% for test (60% remains for training)
        random_state=42
    )

    # all_files = data_manager.get_train_files() + data_manager.get_val_files() + data_manager.get_test_files()
    # all_soap_columns = determine_all_soap_columns(all_files)
    
    # Initialize trainer
    trainer = MOFSetTransformerTrainer(
        data_manager=data_manager,
        aggregator_params={
            'num_seed_points': 1,        # Current: 3, try more seeds
            'num_encoder_blocks': 1,     # Current: 2, try more layers
            'num_decoder_blocks': 1,     # Current: 1, try more
            'heads': 4,                  # Current: 4, try more attention heads
            'dropout': 0.3,              # Current: 0.1, try higher dropout
        },
        loss_metric='mae' 
    )

    trainer.determine_all_soap_columns()
    
    results = trainer.train(
        epochs=500,
        learning_rate=0.00001,  # Lower LR for end-to-end training
        patience=100,
        batch_size=100,      # Process 100 files per epoch
        verbose=True
    )

    trainer.plot_training_history()

    # Save both models
    torch.save({
        'encoder_decoder': results['encoder_decoder'].state_dict(),
        # 'property_predictor': results['property_predictor'].state_dict(),
    }, 'optimized_models.pth')

    return trainer, results

if __name__ == "__main__":
    trainer, results = main()