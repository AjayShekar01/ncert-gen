import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available 
model_id = "google/gemma-7b-it"
# model_id = model_id # (we already set this above)
print(f"[INFO] Using model_id: {model_id}")

# from huggingface_hub import login
# login()

use_quantization_config=False
# 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)