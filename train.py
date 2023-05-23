import torch
from torch.utils.data import Dataset
from transformers import FunnelForMaskedLM, FunnelTokenizerFast, Trainer, TrainingArguments
import os

# Assuming you have your training and validation data in txt format
train_path = 'babylm_data/babylm_10M'
val_path = 'babylm_data/babylm_dev'

# Instantiate the model and tokenizer
model = FunnelForMaskedLM.from_pretrained("funnel-transformer/small")
tokenizer = FunnelTokenizerFast.from_pretrained("funnel-transformer/small")



# Define your training and validation datasets
class MyDataset(Dataset):
    def __init__(self, path):
        self.lines=[]
        for filename in os.scandir(path):
            if filename.is_file():
                self.lines.append(open(filename,errors='replace').readlines())

    def __getitem__(self, idx):
        encoding = tokenizer(self.lines[idx], truncation=True, padding='max_length', return_tensors='pt')
        return {key: torch.squeeze(val) for key, val in encoding.items()}

    def __len__(self):
        return len(self.lines)

train_dataset = MyDataset(train_path)
val_dataset = MyDataset(val_path)

# Define your training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5, # update to a suitable value for your use case
    per_device_train_batch_size=50, # update to suit your available resources
    per_device_eval_batch_size=50, # update to suit your available resources
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()
