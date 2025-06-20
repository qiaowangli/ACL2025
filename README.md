# Optimizing Human Genomics Representation via Contrastive Learning

This repository contains some of the code and pre-trained models used in the project *"Optimizing Human Genomics Representation via Contrastive Learning"*.

**Note:** This document is continuously being updated — more scripts and models will be uploaded soon.

## Project Structure

- `code/`: Contains scripts and modules for:
  - **cl_normal.py**: A sample script for continued training of genomic transformer models on target datasets
  - **train.py**: Supervised training scripts for downstream classification tasks.
  - **run_dnabert**: A bash file for Fine-tuning DNABERT.
  - **run_dnabert2**: A bash file for Fine-tuning DNABERT-2.
  - **run_dnaberts**: A bash file for Fine-tuning DNABERT-S.
  - **Requirements.txt**: A `requirements.txt` file listing all necessary packages for running the codebase.

- `trained_model/`: Includes some of the selected pre-trained model checkpoints:
  - `dnabert_tf.pt`: An optimized version of DNABERT further pre-trained on TF-binding datasets.
  - `dnabert2_tf.pt`: An optimized version of DNABERT-2 further pre-trained on TF-binding datasets.
  - `dnaberts_tf.pt`: An optimized version of DNABERT-S further pre-trained on TF-binding datasets.
## Requirements

- Python 3.8+
- PyTorch >= 1.10
- Transformers (Hugging Face)
- NumPy
- pandas
- scikit-learn
- tqdm
- matplotlib
- seaborn

To install the dependencies, run:

```bash
pip install -r code/requirements.txt
```

To apply further pre-trainning, run:

```bash
nohup python3 cl_normal.py > log/cl_normal_256.log 2>&1
# You need to config the path to store the trained model
```

To apply fine-tuning, run:

```bash
 nohup sh run_dnaberts.sh /Path to your model/pytorch_model.bin  folder_name > log_name.log 2>&1
```