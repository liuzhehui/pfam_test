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
git clone https://github.com/liuzhehui/pfam_test.git
cd pfam_test
```

#### 2. Create and Activate the Conda Environment
```bash
conda env create -f environment.yml
conda activate pfam
```


#### 3. Install Python Packages with Pip

```bash
pip install -r requirements.txt
```

#### 4. Install Jupyter Kernel

If you want to use Jupyter notebooks, install the Jupyter kernel:

```bash
python -m ipykernel install --user --name pfam --display-name "Python (pfam)"
```

## Usage

### Data Preparation

Make sure your data is organized as follows:
data/
├── random_split/
│   ├── train/
│   ├── dev/
│   └── test/

#### Note: The data is not directly provided, but can be downloaded at https://www.kaggle.com/datasets/googleai/pfam-seed-random-split

### Run the Code

To train and evaluate the model, you can simply run the provided Jupyter notebook `./notebook/pfam_classification.ipynb` by following the instructions inside. 
The notebook utilizes custom-written modules located in the `./src` folder, which include various `.py` files necessary for the training and evaluation process.

These are the modules in the './src' folder:

- `data_loading.py`: Contains functions for loading the protein sequences data.
- `data_plotting.py`: Contains functions for data visualisation.
- `cnn.py`: Defines the ProtCNN model architecture for protein family classification.
- `lstm.py`: Defines the Bi-LSTM model architecture for protein family classification.
- `model_train.py`: Script for training the model.
- `evaluate.py`: Contains functions for evaluating the trained model on a test dataset.
- `seq_encoding.py`: Contains functions for encoding the input protein sequence and target labels.

### Results and analysis
- The trained models are saved in the `./models` folder.
- The data table and plots are saved in the `./analysis` folder.
- The full written report is saved in the `./report` folder.

## Hardware that I run my code

- OS: WSL2 - Ubuntu 22.04.2 LTS
- CPU: AMD Ryzen 5 3600 - 6 core 12 threads
- GPU: RTX 4070
- Memory: 32GB
- IDE: Visual Studio Code 

## References

In addition to the reference I listed in the report, the following links helped me guide through the project and provided inspirations:
- https://www.kaggle.com/code/nistugua/pfam-analysis-classification
- https://github.com/ronakvijay/Protein_Sequence_Classification/tree/master
- https://towardsdatascience.com/protein-sequence-classification-99c80d0ad2df