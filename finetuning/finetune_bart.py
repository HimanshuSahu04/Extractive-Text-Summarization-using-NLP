from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import json
import torch
print("CUDA Available:", torch.cuda.is_available())


# -------- Load Data -------- #
with open("data.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# -------- Model & Tokenizer -------- #
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# -------- Tokenization -------- #
def preprocess(batch):
    inputs = tokenizer(
        batch["article"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    outputs = tokenizer(
        batch["summary"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_data = dataset.map(preprocess, batched=True)

# -------- Training Args -------- #
training_args = TrainingArguments(
    output_dir="./bart_finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    fp16=torch.cuda.is_available(),   # 🚀 GPU optimization
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,
    report_to="none"
)


# -------- Trainer -------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data
)

trainer.train()

model.save_pretrained("./bart_finetuned")
tokenizer.save_pretrained("./bart_finetuned")
