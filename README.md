# PFAM Protein Family Prediction

This repository contains code for predicting protein families from sequences using deep learning models.

## Setup

### Prerequisites

Make sure you have the following installed:
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Create Conda Environment

To create a conda environment and install the required dependencies, follow these steps:

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pfam-protein-family-prediction.git
cd pfam-protein-family-prediction

2. Create and Activate the Conda Environment
conda env create -f environment.yml
conda activate pfam

3. Install Python Packages with Pip
pip install -r requirements.txt

4. Install Jupyter Kernel
If you want to use Jupyter notebooks, install the Jupyter kernel:
python -m ipykernel install --user --name pfam --display-name "Python (pfam)"
```

Usage
Data Preparation
Make sure your data is organized as follows:
data/
├── random_split/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv

Run the Code
To train and evaluate the model, run:
python train.py


Example Code
Here’s a brief description of the key files and scripts:

train.py: Script to load data, preprocess, train, and evaluate the model.
environment.yml: Conda environment configuration file.
data/: Directory containing the training, development, and test datasets.
Data Loading
The data is loaded from CSV files located in the data/random_split directory. Each CSV file should have a column named sequence for the protein sequences and a column named family for the labels.

One-Hot Encoding
Sequences are one-hot encoded using OneHotEncoder from scikit-learn. Each amino acid in the sequence is encoded as a binary vector.

Model Training
The model is trained using PyTorch. A simple CNN architecture is used for demonstration purposes. Modify the architecture as needed.


## Hardware that I run my code

OS: WSL2 - Ubuntu 22.04.2 LTS
CPU: AMD Ryzen 5 3600 - 6 core 12 threads
GPU: RTX 4070
Memory: 32GB
IDE: Visual Studio Code 
