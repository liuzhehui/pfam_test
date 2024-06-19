import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sequence_length_distribution(partition, output_dir, title='Sequence Length Distribution'):
    colors = {'train': 'blue', 'dev': 'green', 'test': 'red'}
    
    for key in partition.keys():
        plt.figure(figsize=(12, 6))
        sns.histplot(partition[key]['seq_length'], bins=50, color=colors.get(key, 'blue'))
        plt.title(f'{title} - {key.capitalize()}')
        plt.xlabel('Sequence Length')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{key}_sequence_length_distribution.png', dpi=300)
        plt.close()

def plot_protein_family_distribution(partition, output_dir, title='Protein Family Distribution'):
    colors = {'train': 'blue', 'dev': 'green', 'test': 'red'}
    
    for key in partition.keys():
        plt.figure(figsize=(12, 6))
        sns.histplot(partition[key]['family_accession'].value_counts(), bins=50, color=colors.get(key, 'blue'))
        plt.title(f'{title} - {key.capitalize()}')
        plt.xlabel('Protein Families')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{key}_protein_family_distribution.png', dpi=300)
        plt.close()

def count_amino_acids(partition, key):
    count_amino_acids = dict()

    for sequence in partition[key]['sequence']:
        for amino_acid in list(sequence):
            if amino_acid not in count_amino_acids.keys():
                count_amino_acids[amino_acid] = 1
            else:
                count_amino_acids[amino_acid] += 1

    # Order the dict
    ordered_count = dict(sorted(count_amino_acids.items(), key=lambda item: item[1], reverse=True))
    return ordered_count

def plot_amino_acid_distribution(partition, key, file, title='Amino Acid Distribution'):
    ordered_count = count_amino_acids(partition, key)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(ordered_count)), list(ordered_count.values()), align='center')
    plt.xticks(range(len(ordered_count)), list(ordered_count.keys()))
    plt.title(title)
    plt.xlabel('Amino Acids')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(file, dpi=300)
    plt.close()