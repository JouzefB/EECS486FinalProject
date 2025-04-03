# compare_gpt2_to_wiki.py

import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from wikipedia_lookup import get_wikipedia_summary

# Environment setup for MPS/CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load models once
generator = pipeline("text-generation", model="gpt2", device=-1)
model = SentenceTransformer("all-MiniLM-L6-v2")

def analyze_person_with_wikipedia(person):
    result_lines = []

    # Step 1: GPT-2 output
    prompt = f"Write a short biography of {person}."
    gpt_output = generator(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]

    # Step 2: Wikipedia summary
    wiki_summary = get_wikipedia_summary(person)

    # Step 3: Semantic comparison
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(wiki_summary, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    # Step 4: Report content
    result_lines.append(f"=== GPT-2 Generated Biography for {person} ===")
    result_lines.append(gpt_output)
    result_lines.append("\n=== Wikipedia Summary ===")
    result_lines.append(wiki_summary)
    result_lines.append(f"\nSemantic Similarity: {similarity:.2f}")

    if similarity < 0.6:
        result_lines.append("⚠️ GPT-2 output is SUSPICIOUS.")
    else:
        result_lines.append("✅ GPT-2 output looks consistent.")

    return "\n".join(result_lines)

def run_wiki_comparison(person):
    from wikipedia_lookup import get_wikipedia_summary  # if needed inside

    generator = pipeline("text-generation", model="gpt2", device=-1)
    gpt_output = generator(f"Write a short biography of {person}.", max_length=200, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]
    wiki_summary = get_wikipedia_summary(person)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(wiki_summary, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    lines = []
    lines.append(f"=== GPT-2 Generated Biography for {person} ===")
    lines.append(gpt_output)
    lines.append("\n=== Wikipedia Summary ===")
    lines.append(wiki_summary)
    lines.append(f"\nSemantic Similarity: {similarity:.2f}")
    lines.append("✅ GPT-2 output looks consistent." if similarity >= 0.6 else "⚠️ GPT-2 output is SUSPICIOUS.")
    return "\n".join(lines)




#if wanna test by ourselves
# if __name__ == "__main__":
#     person = "Marie Curie"  # or any name you'd like to test
#     output = analyze_person_with_wikipedia(person)

#     # Print to console
#     print(output)

#     # Optionally write to a file too
#     with open(f"wiki_comparison_{person.replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
#         f.write(output)
