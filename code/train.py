import os
import csv
import copy
import pandas as pd
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import torch.nn as nn
import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import pickle
from transformers import AutoConfig
import random
import torch.nn.functional as F

from transformers import BertModel, BertConfig, PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification

from torch.cuda.amp import autocast




from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)



# seed = random.randint(1, 40000)

# print(seed)

seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class extraEncoder(nn.Module):
    def __init__(self, input_size, heads, dropout, forward_expansion):
        super(extraEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size, 
                nhead=heads, 
                dim_feedforward=input_size * forward_expansion, 
                dropout=dropout
            ), 
            num_layers=1 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        transformer_out = self.transformer(x.unsqueeze(0))
        return transformer_out.squeeze(0)
    


class SingleLayerTransformer(nn.Module):
    def __init__(self, input_size, embed_size, heads, dropout, forward_expansion, num_classes):
        super(SingleLayerTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=heads, 
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        self.fc_out = nn.Linear(embed_size, num_classes)
        
    def forward(self, x, mask=None):
        x = self.embedding(x).unsqueeze(0)
        x = self.transformer_encoder(x, src_key_padding_mask=mask).squeeze(0)
        return self.fc_out(x)
    
    
    

NN_LENGTH = 768
class RegressionNet(nn.Module):
    def __init__(self,NN_LENGTH, dropout_rate=0.3):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(NN_LENGTH, NN_LENGTH)
        self.fc2 = nn.Linear(NN_LENGTH, NN_LENGTH)
        self.fc3 = nn.Linear(NN_LENGTH, NN_LENGTH)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        print(num_labels)
        # exit(-1)
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        # self.base_model.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/vae_skip_cl_batch64_finetune/dnabert2_final.pt"))
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=0.1)
        
        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, num_labels),
        #     nn.ReLU()
        # )
        
        # self.model_extra = extraEncoder(
        #     input_size=self.config.hidden_size,   
        #     heads=4,                 
        #     dropout=0.1,             
        #     forward_expansion=4    
        # ).to(device)


        # self.model_extra.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/vae_skip_cl_batch128/dnabert2_extra_final9.pt"))
        # self.model_extra.eval() 
        
        # self.model_extra = RegressionNet(NN_LENGTH)
        # self.model_extra.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/model_encoder/simpCL.pth"))
        # self.model_extra.eval()             
        
        # /home/roy/Desktop/thesis/thesis/model_encoder/simpCL.pth
        
        
        # self.classifier = SingleLayerTransformer(
        #     input_size=self.config.hidden_size, 
        #     embed_size=128, 
        #     heads=4, 
        #     dropout=0.1, 
        #     forward_expansion=4, 
        #     num_classes=num_labels
        # )
        
        
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        
        # print(len(input_ids))
        # exit(-1)
        
        # print(attention_mask)
        
        # attention_mask = (input_ids != 3).long()
        # print(attention_mask)
        
        # exit()
        
    
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # outputs = self.model_extra(outputs)
        
        # print(outputs)
        
        token_embeddings = outputs[0]
        
        # print(token_embeddings.shape)

        # Expand the attention_mask to match the shape of token_embeddings (to multiply element-wise)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

        # Multiply the embeddings by the attention mask to zero out padding token embeddings
        masked_embeddings = token_embeddings * attention_mask_expanded

        # Calculate the sum of valid token embeddings
        sum_embeddings = masked_embeddings.sum(dim=1)

        # Calculate the number of non-padding tokens per sequence
        non_pad_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
        
        # Compute the mean by dividing the sum of embeddings by the number of non-padding tokens
        custom_output = sum_embeddings / non_pad_tokens.clamp(min=1e-9)  # Avoid division by zero
        
    
        #################################################################################################################
        embeddings = self.base_model.get_input_embeddings()
        
        embeddings = embeddings(input_ids)
        
        # print(embeddings.shape)
        # # exit(-1)
        
        masked_input_embeddings = embeddings * attention_mask_expanded
        sum_input_embeddings = masked_input_embeddings.sum(dim=1)
        input_layer_mean = sum_input_embeddings / non_pad_tokens.clamp(min=1e-9)
        
        # print(input_layer_mean.shape)
        # exit(-1)
        
        # custom_output =  torch.abs(custom_output - input_layer_mean)
        # custom_output = torch.log1p(custom_output)
        
        # exit(-1)
        ##################################################################################################################
        # outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # exit(-1)
        # custom_output = outputs[1]
        
        ##################################################################################################################
        custom_output = self.dropout(custom_output)
        
        # print(custom_output[0])
        # exit(-1)        
        custom_output = custom_output
        
        # custom_output = self.model_extra(custom_output)

        logits = self.classifier(custom_output)
        

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
        
            


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    model_path: Optional[str] = field(default="lol")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=3)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa




"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        # if kmer != -1:
        #     # only write file on the first process
        #     if torch.distributed.get_rank() not in [0, -1]:
        #         torch.distributed.barrier()

        #     logging.warning(f"Using {kmer}-mer as input...")
        #     texts = load_or_generate_kmer(data_path, texts, kmer)

        #     if torch.distributed.get_rank() == 0:
        #         torch.distributed.barrier()
        if kmer != -1:
            # Log the k-mer usage and load/generate it without distributed rank checks
            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)


        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



def transform_and_normalize(vecs, kernel, bias):
    
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
        
    return vecs
    # return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
            
    
"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
        
        
    # # Load the full train.csv file
    # train_data_path = os.path.join(data_args.data_path, "train.csv")
    # train_df = pd.read_csv(train_data_path)

    # # Keep only 1/10th of the rows
    # train_df_reduced = train_df.sample(frac=0.05, random_state=42).reset_index(drop=True)

    # # Save the reduced dataframe to a temporary CSV file
    # reduced_data_path = os.path.join(data_args.data_path, "train_reduced.csv")
    # train_df_reduced.to_csv(reduced_data_path, index=False)

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # load model
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     num_labels=train_dataset.num_labels,
    #     trust_remote_code=True,
    # )

    model = CustomModel(model_args.model_name_or_path, train_dataset.num_labels)
    

    # model.base_model.load_state_dict(torch.load(model_args.model_path))
    # def reinitialize_all_weights(model):
    #     for module in model.modules():
    #         if hasattr(module, 'reset_parameters'):
    #             module.reset_parameters()

    # reinitialize_all_weights(model)


   
    # model.base_model.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/DNABERT_2/pytorch_model.bin"))
    
    # model.base_model.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/test_pure_neg/dnabert2_final0.pt"))

    
    
   
    # model.base_model.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/vae_skip_cl_batch64_finetune_sampling_bert1_tf_1k/dnabert2_final4.pt"))
    # model.base_model.load_state_dict(torch.load("/home/roy/Desktop/thesis/thesis/vae_skip_cl_batch64_finetune_sampling_tf_3k_prom_core_300/dnabert2_final9.pt"))
    
   

    
    
    
    # print(model)
    # print(model_1)
    # # print(model.classifier)
    # exit(-1)

    

    
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # for param in model.model_extra.parameters():
    #     param.requires_grad = False
    
    # for param in model.classifier.parameters():
    #     param.requires_grad = False

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        print(results)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()