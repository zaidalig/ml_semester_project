from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import os

# Load your text dataset
data_files = {
    "train": "data/train.txt",
    "validation": "data/valid.txt"
}
dataset = load_dataset("text", data_files=data_files)

# Load tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # To avoid padding error

model = GPT2LMHeadModel.from_pretrained("gpt2")


# Tokenization with labels
def tokenize_function(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    result["labels"] = result["input_ids"].copy()  # Required for computing loss
    return result


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./ml_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",  # replaces deprecated evaluation_strategy
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Start training
trainer.train()

# Save trained model
model.save_pretrained("ml_model")
tokenizer.save_pretrained("ml_model")

print("✅ Model trained and saved in /ml_model")
