# Google_Knowledge_API_Graph_vs_chatgpt2.py

import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_knowledge_graph_data(query, api_key):
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key,
        "limit": 5,
        "indent": True,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data.get("itemListElement", []):
            result = item["result"]
            if "Person" in result.get("@type", []):
                summary = result.get("name", "") + ". "
                summary += result.get("description", "") + ". "
                summary += result.get("detailedDescription", {}).get("articleBody", "")
                return summary.strip()
    return None

def summarize_differences(kg_text, gpt_text):
    kg_words = set(kg_text.lower().split())
    gpt_words = set(gpt_text.lower().split())
    missing_words = kg_words - gpt_words
    return [w for w in missing_words if len(w) > 4 and w.isalpha()][:10]

def run_kg_comparison(person, api_key, model_name="gpt2"):
    lines = []

    kg_text = get_knowledge_graph_data(person, api_key)
    if kg_text:
        lines.append(f"=== Knowledge Graph Data for {person} ===")
        
        lines.append(kg_text)
    else:
        lines.append(f"No data found in Knowledge Graph for {person}")
        kg_text = ""

    generator = pipeline("text-generation", model=model_name, device=-1)
    gpt_output = generator(f"Write a short biography of {person}.", max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    lines.append(f"\n=== GPT-2 Generated Biography ===")

    lines.append(gpt_output)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_kg = model.encode(kg_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_kg).item()

    lines.append(f"\nSemantic Similarity to Knowledge Graph Data: {similarity:.2f}")
    lines.append("✅ GPT-2 output looks consistent with Knowledge Graph data." if similarity >= 0.6 else "⚠️ GPT-2 output is SUSPICIOUS.")

    lines.append("\n=== Summary of Differences (Missing Key Terms) ===")
    if kg_text:
        missing_keywords = summarize_differences(kg_text, gpt_output)
        if missing_keywords:
            for word in missing_keywords:
                lines.append(f"❌ MISSING KEYWORD: {word}")
        else:
            lines.append("✅ All key terms from Knowledge Graph were referenced.")

    return "\n".join(lines)
