from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Load tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inference helper
def generate_response(model, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Get text after <|assistant|>
    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>\n")[1].strip()
    else:
        response = full_output.strip()

    # Stop at first newline or period if needed
    for stop_token in ["\n", ".", "!", "?"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()
            break

    return response

# Prompt
instruction = "Where is the capital of France?"
prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(model_id)

# Load LoRA-adapted model
lora_model = AutoModelForCausalLM.from_pretrained(model_id)
lora_model = PeftModel.from_pretrained(lora_model, "./tinyllama-lora")

# Generate responses
base_response = generate_response(base_model, prompt)
lora_response = generate_response(lora_model, prompt)

# Print results
print("=== Base Model Response ===")
print("Question: ",instruction)
print("Response: ",base_response)
print("\n=== LoRA-Finetuned Model Response ===")
print("Question: ",instruction)
print("Response: ",lora_response)
