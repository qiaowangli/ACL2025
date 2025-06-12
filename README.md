# Optimizing Human Genomics Representation via Contrastive Learning

This repository contains some of the code and pre-trained models used in the project *"Optimizing Human Genomics Representation via Contrastive Learning"*.
## Project Structure

- `code/`: Contains scripts and modules for:
  - **Further_Pre_training.py**: Continued training of genomic transformer models (e.g., DNABERT, DNABERT-2, DNABERT-S) on target datasets such as domain-specific and public hg38 data.
  - **Fine_tuning.py**: Supervised training scripts for downstream classification tasks.
  - **Evaluation**: A bash file for Fine-tuning.
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

<!-- To apply further pre-trainning, run:

```bash
pip install -r code/requirements.txt
``` -->