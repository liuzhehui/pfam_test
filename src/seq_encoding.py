import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder

# map amino acids to integers

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
rare_amino_acids = 'BJOUXZ'
aa_dict = {aa: i + 1 for i, aa in enumerate(amino_acids)}
for i, aa in enumerate(rare_amino_acids):
    aa_dict[aa] = 0

# encode sequences

def encode_sequence(sequence, max_seq_length):
    """
    Encode a protein sequence into a fixed-length numerical format.
    
    Parameters:
    sequence (str): The protein sequence to encode.
    max_seq_length (int): The maximum sequence length for padding.
    
    Returns:
    list: Encoded sequence with padding.
    """
    sequence_encoded = [aa_dict.get(aa, len(aa_dict)) for aa in sequence[:max_seq_length]]
    sequence_encoded += [0] * (max_seq_length - len(sequence_encoded))
    return sequence_encoded

def encode_partition_sequences(partition, max_seq_length):
    """
    Encode sequences in a partition dictionary.
    
    Parameters:
    partition (dict): Dictionary containing sequences to encode.
    max_seq_length (int): The maximum sequence length for padding.
    
    Returns:
    dict: Partition dictionary with encoded sequences.
    """
    for key in partition.keys():
        partition[key]['encoded_sequence'] = partition[key]['sequence'].apply(
            lambda seq: encode_sequence(seq, max_seq_length))
        partition[key]['sequence_tensor'] = partition[key]['encoded_sequence'].apply(
            lambda x: torch.tensor(x))
    return partition

def pad_partition_sequences(partition):
    """
    Pad encoded sequences in a partition dictionary.
    
    Parameters:
    partition (dict): Dictionary containing encoded sequences.
    
    Returns:
    tuple: Padded tensors for train, dev, and test sets.
    """
    x_train = pad_sequence(partition['train']['sequence_tensor'], batch_first=True)
    x_dev = pad_sequence(partition['dev']['sequence_tensor'], batch_first=True)
    x_test = pad_sequence(partition['test']['sequence_tensor'], batch_first=True)
    return x_train, x_dev, x_test

# one-hot encode labels
def one_hot_encode_labels(partition, column='family_accession'):
    """
    One-hot encode labels in a partition dictionary.
    
    Parameters:
    partition (dict): Dictionary containing labels to encode.
    column (str): Column name of the labels.
    
    Returns:
    tuple: One-hot encoded labels for train, dev, and test sets.
    """
    ohe = OneHotEncoder(sparse=False)
    y_train = ohe.fit_transform(partition['train'][[column]])
    y_dev = ohe.fit_transform(partition['dev'][[column]])
    y_test = ohe.fit_transform(partition['test'][[column]])
    return y_train, y_dev, y_test