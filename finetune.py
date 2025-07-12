import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


# Load tokenizer and base model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Setup LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Load example dataset from file
with open("concise_dataset.json", "r") as f:
    examples = json.load(f)

# Format examples
def format_example(example):
    return f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"

formatted_data = [format_example(e) for e in examples]
# Tokenize dataset
tokenized = tokenizer(
    formatted_data,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)

dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"].tolist(),
    "attention_mask": tokenized["attention_mask"].tolist(),
    "labels": tokenized["input_ids"].tolist()
})

# Split into train and eval
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=10,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained("./tinyllama-lora")
tokenizer.save_pretrained("./tinyllama-lora")

