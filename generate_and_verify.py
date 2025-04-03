#Step 1: Generate a biography using GPT-2
from transformers import pipeline

#Create the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

#Choose a person for your test case
person = "Ada Lovelace"
prompt = f"Write a short biography of {person}."

#Generate the biography
output = generator(prompt, max_length=200, do_sample=True)[0]["generated_text"]

print("=== Generated Biography ===")
print(output)