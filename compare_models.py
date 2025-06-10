from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Model ID
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft_model_path = "./tinyllama-lora"

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prompt to test
# instruction = "What is the capital of France?"
instruction = "Who wrote 1984"
prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
print("\n🔹 Base model output:")
base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
base_out = base_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
print(base_out[0]["generated_text"].replace(prompt, "").strip())

# Load fine-tuned model with LoRA
print("\n🔸 LoRA fine-tuned model output:")
lora_model = AutoModelForCausalLM.from_pretrained(model_id)
lora_model = PeftModel.from_pretrained(lora_model, peft_model_path).to(device)
lora_pipe = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
lora_out = lora_pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
print(lora_out[0]["generated_text"].replace(prompt, "").strip())

