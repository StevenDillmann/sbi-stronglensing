import torch
import h5py
import pandas as pd
import os

def load_csv_labels(file_path: str, keep_columns: list) -> torch.Tensor:
    """Loads parameter labels from a CSV file and converts them to a tensor."""
    df = pd.read_csv(file_path)
    return torch.tensor(df[keep_columns].values, dtype=torch.float32)

def load_hdf5_data(file_path: str, key: str = "data") -> torch.Tensor:
    """Loads image data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]  # Reads all data under the specified key
    return torch.tensor(data, dtype=torch.float32)


def load_data(data_directory, keep_columns):
    labels_path = os.path.join(data_directory, "metadata.csv")
    data_path = os.path.join(data_directory, "image_data.h5")
    labels = load_csv_labels(labels_path, keep_columns)
    data = load_hdf5_data(data_path)
    return labels, data