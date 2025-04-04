# compare_gpt2_to_wiki.py
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from wikipedia_lookup import get_wikipedia_summary

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def run_wiki_comparison(person, model_name="gpt2"):
    lines = []

    # GPT Output
    generator = pipeline("text-generation", model=model_name, device=-1)
    prompt = f"Write a short biography of {person}."
    gpt_output = generator(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]
    
    # Wikipedia
    wiki_summary = get_wikipedia_summary(person)

    # Semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(wiki_summary, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    lines.append(f"=== GPT-2 Generated Biography for {person} ===")
    lines.append(gpt_output)
    lines.append("\n=== Wikipedia Summary ===")
    lines.append(wiki_summary)
    lines.append(f"\nSemantic Similarity: {similarity:.2f}")
    lines.append("✅ GPT-2 output looks consistent." if similarity >= 0.6 else "⚠️ GPT-2 output is SUSPICIOUS.")

    return "\n".join(lines)
