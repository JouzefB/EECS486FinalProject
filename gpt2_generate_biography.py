###trying to test gpt biography generation and how it works



# gpt2_biography_generator.py

from transformers import pipeline

# Initialize the GPT-2 text generation pipeline using CPU

generator = pipeline("text-generation", model="gpt2", device=-1)

# Choose a person for the biography

person = "Ada Lovelace"
prompt = f"Write a short biography of {person}."

# Generate the biography
output = generator(
    prompt,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    truncation=True,
    num_return_sequences=1
)[0]["generated_text"]

# Print the result
print("=== Generated Biography ===")
print(output)
