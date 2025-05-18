import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths and config
train_data_path = "../data/new_train.csv"
test_data_path = "../data/new_test.csv"
model_checkpoint = "answerdotai/ModernBERT-base"
output_dir = "./Base-Bert-finetuned"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
val_df, test_df = train_test_split(test_df, test_size=2/3, random_state=42)

num_labels = train_df['label'].nunique()
print(f"Detected number of classes: {num_labels}")

# Save temp CSVs
train_df.to_csv("train_temp.csv", index=False)
val_df.to_csv("val_temp.csv", index=False)
test_df.to_csv("test_temp.csv", index=False)

# Load with HuggingFace datasets
train_dataset = load_dataset("csv", data_files="train_temp.csv", split="train")
val_dataset = load_dataset("csv", data_files="val_temp.csv", split="train")
test_dataset = load_dataset("csv", data_files="test_temp.csv", split="train")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=False, max_length=512)

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

for ds in [train_dataset, val_dataset, test_dataset]:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Dataloaders
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

scaler = GradScaler(enabled=torch.cuda.is_available())

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_accuracy = 0.0
best_epoch = 0
checkpoint_path = ""

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            val_running_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            val_correct += (preds == batch["labels"]).sum().item()
            val_total += batch["labels"].size(0)

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"\nEpoch {epoch}:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        checkpoint_path = os.path.join(output_dir, f"best_checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved new best model at {checkpoint_path} with Val Acc: {best_val_accuracy:.4f}")

# Load best model for testing
print(f"\n🏆 Best Model from Epoch {best_epoch} with Val Accuracy {best_val_accuracy:.4f}")
print(f"Checkpoint saved at: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Final model save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Test the best model
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        test_correct += (preds == batch["labels"]).sum().item()
        test_total += batch["labels"].size(0)

test_accuracy = test_correct / test_total
print(f"\n🧪 Test Accuracy of Best Model: {test_accuracy:.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label="Train Acc")
plt.plot(range(1, epochs + 1), val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()
