#!/usr/bin/env python3
"""
REMatch kernel SOAP for gas uptake prediction with structure-specific species
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
from ase.io import read
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SOAPGasUptakePredictor:
    def __init__(self, cif_dir, labels_file, cutoff=5.0, l_max=1, n_max=1, 
                 alpha=0.1, gamma=1.0, threshold=0.01):
        """
        Initialize the SOAP gas uptake predictor with REMatch kernel
        """
        self.cif_dir = cif_dir
        self.labels_file = labels_file
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        
        self.rematch_kernel = None
        self.model = None
        self.scaler = None
        
    def load_data(self):
        """Load CIF files and corresponding labels"""
        print("Loading data...")
        
        # Load labels
        self.labels_df = pd.read_csv(self.labels_file)
        print(f"Loaded {len(self.labels_df)} labels")
        
        # Get all CIF files
        cif_files = [f for f in os.listdir(self.cif_dir) if f.endswith('.cif')]
        print(f"Found {len(cif_files)} CIF files")
        
        # Match CIF files with labels
        structures = []
        labels = []
        ids = []
        
        print("Matching CIF files with labels...")
        for _, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df), desc="Loading structures"):
            cif_id = str(row['id']).strip()  # Remove leading/trailing whitespace
            cif_file = f"{cif_id}.cif"
            cif_path = os.path.join(self.cif_dir, cif_file)
            
            if os.path.exists(cif_path):
                try:
                    structure = read(cif_path)
                    structures.append(structure)
                    # Take the first non-id column as gas uptake
                    uptake_columns = [col for col in self.labels_df.columns if col != 'id']
                    if uptake_columns:
                        labels.append(row[uptake_columns[0]])
                    else:
                        raise ValueError("No gas uptake column found in labels file")
                    ids.append(cif_id)
                except Exception as e:
                    print(f"Error reading {cif_file}: {e}")
                    continue
            else:
                print(f"CIF file not found: {cif_file}")
                continue
        
        self.structures = structures
        self.labels = np.array(labels)
        self.ids = ids
        
        print(f"Successfully loaded {len(self.structures)} structures with labels")
        print(f"Gas uptake range: {self.labels.min():.3f} to {self.labels.max():.3f}")
        
        return self.structures, self.labels, self.ids
    
    def compute_soap_descriptors(self):
        """Compute SOAP descriptors for each structure using only its own species"""
        print("Computing structure-specific SOAP descriptors...")
        
        self.structure_descriptors = []
        self.structure_species = []
        self.soap_descriptor_objects = []
        
        for i, structure in enumerate(tqdm(self.structures, desc="Computing SOAP descriptors")):
            try:
                # Get unique species for this specific structure
                structure_species = sorted(list(set(structure.get_chemical_symbols())))
                
                # Create SOAP descriptor for this structure's species only
                soap_desc = SOAP(
                    species=structure_species,
                    periodic=True,
                    r_cut=self.cutoff,
                    n_max=self.n_max,
                    l_max=self.l_max,
                    sigma=1.0,
                    rbf="gto",
                    sparse=False
                )
                
                # Compute SOAP descriptor
                soap_features = soap_desc.create(structure)
                
                # Normalize features for stability (important for REMatch)
                soap_features_normalized = normalize(soap_features)
                
                # Store everything we need
                self.structure_descriptors.append(soap_features_normalized)
                self.structure_species.append(structure_species)
                self.soap_descriptor_objects.append(soap_desc)
                
            except Exception as e:
                print(f"Error computing SOAP for structure {i}: {e}")
                # Use fallbacks
                self.structure_descriptors.append(np.array([]))
                self.structure_species.append([])
                self.soap_descriptor_objects.append(None)
        
        print(f"Computed descriptors for {len(self.structure_descriptors)} structures")
        print("Each structure uses only its own species - memory efficient!")
        
        return self.structure_descriptors
    
    def get_species_pairs(self, species):
        """Get all species pairs for a given species list"""
        pairs = []
        for i, sp1 in enumerate(species):
            for j, sp2 in enumerate(species):
                if i <= j:  # Avoid duplicates
                    pairs.append((sp1, sp2))
        return pairs
    
    def align_soap_by_pairs(self, soap_desc, soap_obj, original_pairs, target_pairs):
        """
        Align SOAP features by species pairs using get_location()
        """
        if len(soap_desc) == 0 or soap_obj is None:
            # Return zeros for target pairs
            estimated_features_per_pair = 10  # Rough estimate based on nmax, lmax
            return np.zeros(len(target_pairs) * estimated_features_per_pair)
        
        if set(original_pairs) == set(target_pairs):
            # Average over atoms and return 1D array
            avg_soap_desc = np.mean(soap_desc, axis=0)
            return avg_soap_desc
        
        # Average over atoms first to get structure-level descriptor
        avg_soap_desc = np.mean(soap_desc, axis=0)
        
        # Get locations for original pairs
        pair_locations = {}
        for pair in original_pairs:
            try:
                location = soap_obj.get_location(pair)
                # Handle slice object
                if isinstance(location, slice):
                    start = location.start if location.start is not None else 0
                    stop = location.stop if location.stop is not None else len(avg_soap_desc)
                    step = location.step if location.step is not None else 1
                    indices = list(range(start, stop, step))
                else:
                    indices = list(location) if hasattr(location, '__iter__') else [location]
                pair_locations[pair] = indices
            except Exception as e:
                continue  # Skip problematic pairs
        
        # Estimate features per pair
        if len(original_pairs) > 0 and len(avg_soap_desc) > 0:
            features_per_pair = len(avg_soap_desc) // len(original_pairs)
        else:
            features_per_pair = 10  # Default estimate
        
        # Create aligned descriptor
        aligned_desc = np.zeros(len(target_pairs) * features_per_pair)
        
        # Fill in features for pairs that exist
        for i, pair in enumerate(target_pairs):
            if pair in pair_locations:
                indices = pair_locations[pair]
                start_new = i * features_per_pair
                
                # Copy available features
                for j, idx in enumerate(indices):
                    if idx < len(avg_soap_desc) and start_new + j < len(aligned_desc):
                        aligned_desc[start_new + j] = avg_soap_desc[idx]
        
        return aligned_desc
    
    def create_aligned_descriptors_for_rematch(self, indices, global_pairs=None):
        """Create aligned descriptors for a set of structures to use with REMatch"""
        aligned_descriptors = []
        
        if global_pairs is None:
            # Get all unique species pairs across these structures
            all_pairs = set()
            for idx in indices:
                species = self.structure_species[idx]
                pairs = self.get_species_pairs(species)
                all_pairs.update(pairs)
            all_pairs = sorted(all_pairs)
        else:
            all_pairs = global_pairs
        
        print(f"Aligning {len(indices)} structures to {len(all_pairs)} species pairs")
        
        # Align each structure's descriptor to the common species pair space
        for idx in tqdm(indices, desc="Aligning descriptors for REMatch"):
            soap_desc = self.structure_descriptors[idx]
            soap_obj = self.soap_descriptor_objects[idx]
            species = self.structure_species[idx]
            original_pairs = self.get_species_pairs(species)
            
            aligned_desc = self.align_soap_by_pairs(soap_desc, soap_obj, original_pairs, all_pairs)
            # REMatch expects each structure as a 2D array (n_atoms, n_features)
            # But we've averaged over atoms, so we have 1D. Make it 2D with 1 row
            aligned_descriptors.append(aligned_desc.reshape(1, -1))
        
        return aligned_descriptors, all_pairs
    
    def setup_rematch_kernel(self):
        """Setup REMatch kernel"""
        print("Setting up REMatch kernel...")
        
        self.rematch_kernel = REMatchKernel(
            metric="rbf",  # or "linear", "polynomial"
            gamma=self.gamma,
            alpha=1.0,
            threshold=self.threshold
        )
        
        print("REMatch kernel configured")
        return self.rematch_kernel
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"Splitting data (test_size={test_size})...")
        
        indices = np.arange(len(self.structures))
        
        self.train_idx, self.test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Split labels
        self.train_labels = self.labels[self.train_idx]
        self.test_labels = self.labels[self.test_idx]
        
        print(f"Training set: {len(self.train_idx)} samples")
        print(f"Test set: {len(self.test_idx)} samples")
        
        return self.train_idx, self.test_idx
    
    def train_model(self):
        """Train the model using REMatch kernel with aligned descriptors"""
        print("Training model with REMatch kernel...")
        
        # First, get global species pairs from ALL data (train + test)
        print("Finding global species pairs across all data...")
        all_indices = list(self.train_idx) + list(self.test_idx)
        global_pairs = set()
        for idx in all_indices:
            species = self.structure_species[idx]
            pairs = self.get_species_pairs(species)
            global_pairs.update(pairs)
        self.global_pairs = sorted(global_pairs)
        print(f"Found {len(self.global_pairs)} unique species pairs across all data")
        
        # Create aligned descriptors for training data using global pairs
        print("Preparing aligned descriptors for training...")
        train_aligned_descriptors, _ = self.create_aligned_descriptors_for_rematch(
            self.train_idx, self.global_pairs
        )
        
        # Compute REMatch kernel matrix
        print("Computing REMatch kernel matrix...")
        
        # Debug: check shapes
        print(f"Number of training structures: {len(train_aligned_descriptors)}")
        if len(train_aligned_descriptors) > 0:
            print(f"First descriptor shape: {train_aligned_descriptors[0].shape}")
            print(f"First descriptor type: {type(train_aligned_descriptors[0])}")
        
        K_train = self.rematch_kernel.create(train_aligned_descriptors)
        print(f"Training kernel matrix shape: {K_train.shape}")
        
        # Scale labels
        self.scaler = StandardScaler()
        train_labels_scaled = self.scaler.fit_transform(self.train_labels.reshape(-1, 1)).ravel()
        
        # Train kernel ridge regression
        self.model = KernelRidge(
            alpha=self.alpha,
            kernel='precomputed'
        )
        
        print("Fitting model...")
        self.model.fit(K_train, train_labels_scaled)
        
        print("Model training completed!")
        
        # Store training descriptors for prediction
        self.train_aligned_descriptors = train_aligned_descriptors
        
        return self.model
    
    def predict(self):
        """Make predictions on test data"""
        print("Making predictions...")
        
        # Create aligned descriptors for test data using the same global pairs
        print("Preparing aligned descriptors for testing...")
        test_aligned_descriptors, _ = self.create_aligned_descriptors_for_rematch(
            self.test_idx, self.global_pairs
        )
        
        # Compute REMatch kernel matrix between test and train
        print("Computing test kernel matrix...")
        K_test = self.rematch_kernel.create(test_aligned_descriptors, self.train_aligned_descriptors)
        print(f"Test kernel matrix shape: {K_test.shape}")
        
        # Make predictions
        predictions_scaled = self.model.predict(K_test)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        
        return predictions
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Make predictions
        predictions = self.predict()
        
        # Calculate metrics
        mae = mean_absolute_error(self.test_labels, predictions)
        mse = mean_squared_error(self.test_labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.test_labels, predictions)
        
        print(f"\nModel Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Store predictions for plotting
        self.test_predictions = predictions
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'true_values': self.test_labels
        }
    
    def plot_results(self, save_path=None):
        """Plot prediction results"""
        plt.figure(figsize=(10, 8))
        
        # Parity plot
        plt.subplot(2, 2, 1)
        plt.scatter(self.test_labels, self.test_predictions, alpha=0.6)
        plt.plot([self.test_labels.min(), self.test_labels.max()], 
                [self.test_labels.min(), self.test_labels.max()], 'r--', lw=2)
        plt.xlabel('True Gas Uptake')
        plt.ylabel('Predicted Gas Uptake')
        plt.title('Parity Plot')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = self.test_predictions - self.test_labels
        plt.scatter(self.test_labels, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Gas Uptake')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Histogram of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        # Training vs test labels distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.train_labels, bins=30, alpha=0.7, label='Train', edgecolor='black')
        plt.hist(self.test_labels, bins=30, alpha=0.7, label='Test', edgecolor='black')
        plt.xlabel('Gas Uptake')
        plt.ylabel('Frequency')
        plt.title('Data Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def run_full_pipeline(self, test_size=0.2, random_state=42, plot=True):
        """Run the complete pipeline"""
        print("Starting full pipeline...")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Compute structure-specific SOAP descriptors
        self.compute_soap_descriptors()
        
        # Split data
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Setup REMatch kernel
        self.setup_rematch_kernel()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        results = self.evaluate_model()
        
        # Plot results
        if plot:
            self.plot_results()
        
        print("="*50)
        print("Pipeline completed successfully!")
        
        return results

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = SOAPGasUptakePredictor(
        cif_dir="../CIF_files",
        labels_file="../id_labels.csv",
        cutoff=5.0,
        l_max=1,
        n_max=1,
        alpha=0.1,
        gamma=1.0,
        threshold=0.01
    )
    
    # Run full pipeline
    results = predictor.run_full_pipeline(
        test_size=0.2,
        random_state=42,
        plot=True
    )
    
    print(f"\nFinal Results Summary:")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")