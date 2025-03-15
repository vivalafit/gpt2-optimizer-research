import os
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(base_dir, "data", "train_iso.csv"),
            "val": os.path.join(base_dir, "data", "val_iso.csv"),
            "test": os.path.join(base_dir, "data", "test_iso.csv"),
        }
    )
    return dataset

def tokenize_data(dataset, tokenizer):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["requirement"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized = dataset.map(preprocess_function, batched=True)
    return tokenized