import os
from transformers import pipeline

# Force CPU only
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("Device set to use cpu")

generator = pipeline("text-generation", model="gpt2", device=-1)

person = "Ada Lovelace"
prompt = f"Write a short biography of {person}."

output = generator(
    prompt,
    max_length=200,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    num_return_sequences=1
)[0]["generated_text"]

print("=== Generated Biography ===")
print(output)
