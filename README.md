# AGDA-DDI
**AGDA-DDI: Augmented Graph with Drug Attributes for Drug-Drug Interaction
## Project Introduction
This project aims to predict drug-drug interactions (DDIs). It provides various model architectures and uses 5-fold cross-validation and multi-model comparison to effectively predict the types of interactions between drugs by combining drug features (such as chemical similarity, target similarity, etc.) and graph structural information.

## File Structure
* main.py: The main program file for training and testing the model.
* merge_csv.py: A script to merge multiple CSV files and calculate the maximum values.
* model.py: Defines various model architectures, including base models, GAT, GCN, and their variants.
* run_script.py: A script to run experiments in batches.
* train_on_fold.py: Implements 5-fold cross-validation for training and testing.
* utils.py: Provides data processing and graph preprocessing utilities.
* data/: A directory containing data files, including drug interaction data and feature matrices.


## Data Preparation
Data files should be placed in the `data/` directory, including the following:
- `ddi_class_65.csv`: Drug interaction data containing drug pairs and labels.
- `chem_Jacarrd_sim.csv`: Chemical structure similarity matrix of drugs.
- `target_Jacarrd_sim.csv`: Target similarity matrix of drugs.
- `enzyme_Jacarrd_sim.csv`: Enzyme similarity matrix of drugs.
- `pathway_Jacarrd_sim.csv`: Pathway similarity matrix of drugs.

## Environment Dependencies
This project is developed based on Python 3.8+ and depends on the following main libraries:
- torch==2.4.0+cu124
- torch-geometric==2.5.3
- numpy==1.24.1
- pandas==2.0.3
- scikit-learn==1.3.2
- matplotlib==3.7.5

## Installation Method
1. Create and activate a virtual environment (recommended to use Conda):
   ```bash
   conda create -n ddi_env python=3.8
   conda activate ddi_env
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Running Method
### Single Training and Testing
Run the main program, specifying the model type and hyperparameters:
   ```bash
   python Deng_main.py --model 1
   ```

#### Parameter Description

| Parameter Name   | Description                          | Default Value |
|-------------------|--------------------------------------|--------------|
| `--hidden_dim`    | Number of hidden features            | 512          |
| `--out_dim`       | Number of output features            | 256          |
| `--dropout`       | Dropout rate                         | 0.3          |
| `--rels`          | Number of relations                   | 65           |
| `--n_epochs`      | Number of training epochs            | 400          |
| `--batch_size`    | Batch size                           | 1024         |
| `--threshold`     | Edge threshold                       | 0.4          |
| `--model`         | Model type                           | 1            |

**Available Model Types and Corresponding Command Line Parameter Values:**
* Model (Default): --model 1
* Model_avg: --model avg
* Model_sum: --model sum
* Model_wo_feat: --model feat
* Model_wo_diffgraph: --model dg
* Model_wo_relgraph: --model rg
* Model_wo_att: --model att
* Model_GAT: --model gat
* Model_GCN: --model gcn

**Other parameters (such as hidden layer dimensions, number of training epochs, etc.) can be set through command line arguments in main.py.**

### 5-Fold Cross-Validation
Run the 5-fold cross-validation script:
   ```bash
   python Deng_train_on_fold.py
   ```
**This script automatically splits the data into 5 folds and trains and tests the model for each fold, saving the results for each fold.**

### Merge Results
Run the following script to merge the results from multiple CSV files and calculate the maximum values:
   ```bash
   python merge_csv.py
   ```
**Output Results:**
* Training and testing results will be saved to the results.csv file.
* AUCPR and F1 scores for each class will be saved to the class_metrics_fold_k.csv files.
* Cross-validation results will be saved to the cross_validation.csv file.