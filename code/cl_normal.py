import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random


# Set random seed for reproducibility
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==== Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME_2 = "zhihan1996/DNABERT-2-117M"
MODEL_NAME_1 = "zhihan1996/DNA_bert_6"

# ==== Load tokenizers and models ====
tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_NAME_2, trust_remote_code=True)
# tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_NAME_1, trust_remote_code=True)
model_2 = AutoModel.from_pretrained(MODEL_NAME_2, trust_remote_code=True)

# model_2.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/NDCL/result_model/cosin_model_new/final_model/pytorch_model.bin"))

# model_1 = AutoModel.from_pretrained(MODEL_NAME_1, trust_remote_code=True)

# ==== Dataset Class ====
class DNADataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        data = pd.read_csv(csv_file)
        self.sequences = data['sequence'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            **{key: val.squeeze(0) for key, val in encoding.items()},
            "sequence": seq
        }

def custom_collate_fn(batch):
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch] 
    return collated

def cosine_similarity_loss(x, y, device, temp=0.05):

    y_pred = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).reshape(-1, x.shape[1])  # [2N, d]

    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp
    loss = F.cross_entropy(sim, y_true)
    # print(torch.mean(loss))
    
    return torch.mean(loss) * 2

# ==== Dual DNABERT Model ====
class DualDNABERTModel(nn.Module):
    def __init__(self, model2):
        super().__init__()
        # self.model1 = model1
        self.model2 = model2
    def forward(self, input_ids_2, attention_mask_2):

        h2 = self.model2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # shape: [batch_size, seq_len, hidden_dim]
        attention_mask_2 = attention_mask_2.unsqueeze(-1)  # shape: [batch_size, seq_len, 1]
        h2_masked = h2 * attention_mask_2  # zeros out padding positions
        sum_hidden = h2_masked.sum(dim=1)  # shape: [batch_size, hidden_dim]
        valid_token_count = attention_mask_2.sum(dim=1)  # shape: [batch_size, 1]
        valid_token_count = valid_token_count.clamp(min=1)
        mean_hidden = sum_hidden / valid_token_count  # shape: [batch_size, hidden_dim]

        # print(h2.shape)
        # print(mean_hidden.shape)
        # exit(-1)

        return mean_hidden

# ==== Custom Trainer ====
class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device

        input_ids_2 = inputs["input_ids"].to(device)
        attention_mask_2 = inputs["attention_mask"].to(device)


        # Forward through both models
        embed_2 = model(
            input_ids_2=input_ids_2,
            attention_mask_2=attention_mask_2,
        )

        embed_2_variance = model(
            input_ids_2=input_ids_2,
            attention_mask_2=attention_mask_2,
        )


        # print(embed_2[0][0])
        # print(embed_2_variance[0][0])
        # exit(-1)

        loss = cosine_similarity_loss(embed_2, embed_2_variance, device)

        return (loss, SequenceClassifierOutput(loss=loss, logits=embed_2)) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = next(model.parameters()).device
        
        input_ids_2 = inputs["input_ids"].to(device)
        attention_mask_2 = inputs["attention_mask"].to(device)
        
        with torch.no_grad(): 
            # Forward through both models
            embed_2 = model(
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
            )

            embed_2_variance = model(
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
            )

            loss = cosine_similarity_loss(embed_2, embed_2_variance, device)
            
            return loss, embed_2, None 

# ==== Load dataset ====
dataset = DNADataset(
    csv_file="/home/roy/Desktop/thesis/thesis/nn_dna_sequence/all_chromosomes_data.csv",
    tokenizer=tokenizer_2,
    max_length=64
)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ==== Training arguments ====
training_args = TrainingArguments(
    output_dir="/home/roy/Desktop/thesis/thesis/NDCL/result_model/cl_normal_256",
    num_train_epochs=20,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    evaluation_strategy="steps", 
    eval_steps=50,             
    save_strategy="steps",
    learning_rate=5e-5,
    warmup_steps=10,
    logging_steps=50,
    save_steps=50,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    load_best_model_at_end=True, 
    metric_for_best_model="loss",
    greater_is_better=False

)

# ==== Initialize dual model ====
dual_model = DualDNABERTModel(model_2).to(device)

# ==== Trainer ====
trainer = ContrastiveTrainer(
    model=dual_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn
)

# ==== Training and Evaluation ====
print("Starting training...")
trainer.train()
print("Training completed!")

print("Saving model...")
trainer.save_model("/home/roy/Desktop/thesis/thesis/NDCL/result_model/cl_normal_256/best_model")
print("Model saved!")

print("Starting evaluation...")
results = trainer.evaluate()
print("Evaluation results:", results)


print("Saving DNABERT-2 model...")
dual_model.model2.save_pretrained("/home/roy/Desktop/thesis/thesis/NDCL/result_model/cl_normal_256/dnabert2")
tokenizer_2.save_pretrained("/home/roy/Desktop/thesis/thesis/NDCL/result_model/cl_normal_256/dnabert2")
print("DNABERT-2 saved!")
print("All done!")