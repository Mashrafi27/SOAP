import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import pickle



class SOAPDataset(Dataset):
    def __init__(self, soap_data_list, targets):
        self.soap_data_list = soap_data_list
        self.targets = targets

    def __len__(self):
        return len(self.soap_data_list)

    def __getitem__(self, idx):
        soap_tensor = torch.tensor(self.soap_data_list[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return soap_tensor, target


def mil_collate_fn(batch):
    envs, targets = zip(*batch)
    targets = torch.stack(targets)
    padded_envs = pad_sequence(envs, batch_first=True)

    lengths = torch.tensor([e.shape[0] for e in envs])
    B, K_max = len(envs), padded_envs.shape[1]
    mask = torch.zeros(B, K_max, dtype=torch.float)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    return padded_envs, mask, targets


def normalize_filename(filename: str) -> str:
    """
    Normalize filename by removing extension and cleaning up
    """
    # Remove common extensions
    base_name = filename
    extensions = ['.cif', '.CIF', '.xyz', '.XYZ', '.pdb', '.PDB']
    
    for ext in extensions:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            break
    
    return base_name.strip()


def load_soap_matrices(soap_file_path: str = 'soap_2d_matrices.npz', 
                      metadata_file_path: str = 'soap_2d_metadata.pkl') -> Tuple[Dict, Dict]:
    """
    Load SOAP matrices and metadata from saved files
    """
    print("Loading SOAP matrices...")
    
    # Load matrices
    soap_data = np.load(soap_file_path)
    
    # Load metadata
    with open(metadata_file_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {len(soap_data.files)} SOAP matrices")
    print(f"Each matrix has {metadata['n_columns']} features")
    
    return dict(soap_data), metadata


def load_labels(labels_file_path: str = 'ID Labels 1.csv') -> pd.DataFrame:
    """
    Load labels from CSV file
    """
    print("Loading labels...")
    
    labels_df = pd.read_csv(labels_file_path)
    print(f"Loaded {len(labels_df)} labels")
    print(f"Columns: {labels_df.columns.tolist()}")
    
    # Normalize the ID column for matching
    labels_df['normalized_id'] = labels_df['id'].apply(normalize_filename)
    
    return labels_df


def match_soap_with_labels(soap_matrices: Dict, labels_df: pd.DataFrame) -> Tuple[List, List, List]:
    """
    Match SOAP matrices with their corresponding labels
    """
    print("Matching SOAP matrices with labels...")
    
    matched_soap_data = []
    matched_targets = []
    matched_ids = []
    
    unmatched_soap = []
    unmatched_labels = []
    
    # Create a mapping from normalized label IDs to labels
    label_mapping = dict(zip(labels_df['normalized_id'], labels_df['label']))
    
    # Process each SOAP matrix
    for soap_filename, soap_matrix in soap_matrices.items():
        # Normalize the SOAP filename
        normalized_soap_name = normalize_filename(soap_filename)
        
        # Try to find matching label
        if normalized_soap_name in label_mapping:
            matched_soap_data.append(soap_matrix)
            matched_targets.append(label_mapping[normalized_soap_name])
            matched_ids.append(soap_filename)
        else:
            unmatched_soap.append(soap_filename)
    
    # Check for unmatched labels
    soap_normalized_names = {normalize_filename(name) for name in soap_matrices.keys()}
    for _, row in labels_df.iterrows():
        if row['normalized_id'] not in soap_normalized_names:
            unmatched_labels.append(row['id'])
    
    print(f"✓ Successfully matched: {len(matched_soap_data)} samples")
    print(f"⚠ Unmatched SOAP matrices: {len(unmatched_soap)}")
    print(f"⚠ Unmatched labels: {len(unmatched_labels)}")
    
    if unmatched_soap:
        print(f"First few unmatched SOAP files: {unmatched_soap[:5]}")
    
    if unmatched_labels:
        print(f"First few unmatched labels: {unmatched_labels[:5]}")
    
    return matched_soap_data, matched_targets, matched_ids


def create_soap_dataset(soap_file_path: str = 'soap_2d_matrices.npz',
                       metadata_file_path: str = 'soap_2d_metadata.pkl',
                       labels_file_path: str = 'ID Labels 1.csv') -> Tuple[SOAPDataset, List[str]]:
    """
    Create a SOAPDataset from saved SOAP matrices and labels
    """
    # Load SOAP matrices
    soap_matrices, metadata = load_soap_matrices(soap_file_path, metadata_file_path)
    
    # Load labels
    labels_df = load_labels(labels_file_path)
    
    # Match SOAP matrices with labels
    matched_soap_data, matched_targets, matched_ids = match_soap_with_labels(soap_matrices, labels_df)
    
    if len(matched_soap_data) == 0:
        raise ValueError("No matching samples found between SOAP matrices and labels!")
    
    # Create dataset
    dataset = SOAPDataset(matched_soap_data, matched_targets)
    
    print(f"\n=== Dataset Created Successfully ===")
    print(f"Total samples: {len(dataset)}")
    print(f"SOAP matrix shape example: {matched_soap_data[0].shape}")
    print(f"Target range: [{min(matched_targets):.3f}, {max(matched_targets):.3f}]")
    
    return dataset, matched_ids


def analyze_dataset_statistics(dataset: SOAPDataset) -> None:
    """
    Analyze and print dataset statistics
    """
    print("\n=== Dataset Statistics ===")
    
    # Analyze shapes
    shapes = [soap_data.shape for soap_data in dataset.soap_data_list]
    n_atoms = [shape[0] for shape in shapes]
    n_features = [shape[1] for shape in shapes]
    
    print(f"Number of samples: {len(dataset)}")
    print(f"Features per sample: {n_features[0]} (should be consistent)")
    print(f"Atoms per MOF - Min: {min(n_atoms)}, Max: {max(n_atoms)}, Avg: {np.mean(n_atoms):.1f}")
    
    # Analyze targets
    targets = [target.item() if torch.is_tensor(target) else target for target in dataset.targets]
    print(f"Target statistics - Min: {min(targets):.3f}, Max: {max(targets):.3f}, Mean: {np.mean(targets):.3f}, Std: {np.std(targets):.3f}")
    
    # Check for unique feature counts
    unique_features = set(n_features)
    if len(unique_features) == 1:
        print(f"✓ All samples have consistent feature count: {list(unique_features)[0]}")
    else:
        print(f"⚠ Inconsistent feature counts found: {unique_features}")


def create_dataloader(dataset: SOAPDataset, 
                     batch_size: int = 32, 
                     shuffle: bool = True, 
                     num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader with the custom collate function
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=mil_collate_fn,
        num_workers=num_workers
    )
    
    print(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}")
    
    return dataloader


def test_dataloader(dataloader: DataLoader) -> None:
    """
    Test the dataloader by processing one batch
    """
    print("\n=== Testing DataLoader ===")
    
    try:
        batch = next(iter(dataloader))
        padded_envs, mask, targets = batch
        
        print(f"Batch size: {targets.shape[0]}")
        print(f"Padded environments shape: {padded_envs.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"✓ DataLoader working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {str(e)}")
        return False


def split_dataset(dataset: SOAPDataset, 
                 matched_ids: List[str],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42) -> Tuple[SOAPDataset, SOAPDataset, SOAPDataset, Dict]:
    """
    Split dataset into train/validation/test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get total number of samples
    total_samples = len(dataset)
    
    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    print(f"\n=== Dataset Split ===")
    print(f"Total samples: {total_samples}")
    print(f"Train: {train_size} ({train_ratio:.1%})")
    print(f"Validation: {val_size} ({val_ratio:.1%})")
    print(f"Test: {test_size} ({test_ratio:.1%})")
    
    # Create random indices
    indices = torch.randperm(total_samples).tolist()
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_soap_data = [dataset.soap_data_list[i] for i in train_indices]
    train_targets = [dataset.targets[i] for i in train_indices]
    train_ids = [matched_ids[i] for i in train_indices]
    
    val_soap_data = [dataset.soap_data_list[i] for i in val_indices]
    val_targets = [dataset.targets[i] for i in val_indices]
    val_ids = [matched_ids[i] for i in val_indices]
    
    test_soap_data = [dataset.soap_data_list[i] for i in test_indices]
    test_targets = [dataset.targets[i] for i in test_indices]
    test_ids = [matched_ids[i] for i in test_indices]
    
    # Create new dataset objects
    train_dataset = SOAPDataset(train_soap_data, train_targets)
    val_dataset = SOAPDataset(val_soap_data, val_targets)
    test_dataset = SOAPDataset(test_soap_data, test_targets)
    
    # Create split info dictionary
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'random_seed': random_seed,
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    }
    
    return train_dataset, val_dataset, test_dataset, split_info


def create_split_dataloaders(train_dataset: SOAPDataset,
                           val_dataset: SOAPDataset, 
                           test_dataset: SOAPDataset,
                           batch_size: int = 32,
                           num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/validation/test sets
    """
    print(f"\n=== Creating DataLoaders ===")
    print(f"Batch size: {batch_size}")
    
    # Train dataloader (with shuffling)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=mil_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Validation dataloader (no shuffling)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        collate_fn=mil_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Test dataloader (no shuffling)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        collate_fn=mil_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    return train_dataloader, val_dataloader, test_dataloader


def analyze_split_statistics(train_dataset: SOAPDataset, 
                           val_dataset: SOAPDataset, 
                           test_dataset: SOAPDataset) -> None:
    """
    Analyze statistics of each split to ensure they're similar
    """
    print(f"\n=== Split Statistics Analysis ===")
    
    def get_target_stats(dataset, name):
        targets = [target.item() if torch.is_tensor(target) else target for target in dataset.targets]
        return {
            'name': name,
            'count': len(targets),
            'mean': np.mean(targets),
            'std': np.std(targets),
            'min': np.min(targets),
            'max': np.max(targets)
        }
    
    train_stats = get_target_stats(train_dataset, 'Train')
    val_stats = get_target_stats(val_dataset, 'Validation')
    test_stats = get_target_stats(test_dataset, 'Test')
    
    print(f"{'Split':<12} {'Count':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for stats in [train_stats, val_stats, test_stats]:
        print(f"{stats['name']:<12} {stats['count']:<8} {stats['mean']:<10.3f} "
              f"{stats['std']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f}")


def save_split_info(split_info: Dict, filename: str = 'dataset_split_info.pkl') -> None:
    """
    Save split information for reproducibility
    """
    with open(filename, 'wb') as f:
        pickle.dump(split_info, f)
    
    # Also save as text files for easy reference
    with open('train_ids.txt', 'w') as f:
        for soap_id in split_info['train_ids']:
            f.write(f"{soap_id}\n")
    
    with open('val_ids.txt', 'w') as f:
        for soap_id in split_info['val_ids']:
            f.write(f"{soap_id}\n")
    
    with open('test_ids.txt', 'w') as f:
        for soap_id in split_info['test_ids']:
            f.write(f"{soap_id}\n")
    
    print(f"\nSplit info saved to '{filename}' and individual ID files")