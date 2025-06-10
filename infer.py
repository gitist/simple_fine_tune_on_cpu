from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Load fine-tuned model and tokenizer
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft_model_path = "./tinyllama-lora"

tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Move to CPU (or GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Setup pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Prompt
instruction = "Where is the capital of France?"
prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"

# Generate response
output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
print("\n📝 Prompt:")
print(instruction)
print("\n🧠 Response:")
print(output[0]["generated_text"].replace(prompt, "").strip())

