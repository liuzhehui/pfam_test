import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset,DataLoader

# Load data

def load_data(folder):
    data = []
    for file in os.listdir(folder):
        file_data = pd.read_csv(os.path.join(folder, file), index_col=None)
        data.append(file_data.reset_index(drop=True))
    return pd.concat(data, ignore_index=True)
    
# Prepare data for training

def prepare_dataloader(x, y, batch_size, dtype_x=torch.float32, dtype_y=torch.long):
    # Convert one-hot encoded labels to class indices
    y_class_indices = torch.argmax(torch.tensor(y, dtype=dtype_y), dim=1)
    dataset = TensorDataset(torch.tensor(x, dtype=dtype_x), y_class_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

