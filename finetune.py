from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

# Model + tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Enable LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust as needed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Toy dataset
examples = [
    {"instruction": "What is the capital of France?", "response": "Paris."},
    {"instruction": "Who wrote 1984?", "response": "George Orwell."},
    {"instruction": "What's 2 + 2?", "response": "4."},
]

def format_example(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"

formatted_data = [format_example(e) for e in examples]
tokenized = tokenizer(formatted_data, truncation=True, padding=True, return_tensors="pt")

dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "labels": tokenized["input_ids"]
})

# Training args
training_args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,  # We're on CPU
    optim="adamw_torch",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./tinyllama-lora")
tokenizer.save_pretrained("./tinyllama-lora")

