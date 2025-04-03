import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

# Optional MPS fallback for Mac compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_knowledge_graph_data(query, api_key):
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key,
        "limit": 5,  # check a few results, not just top 1
        "indent": True,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data.get("itemListElement", []):
            result = item["result"]
            types = result.get("@type", [])
            if "Person" in types:  # Only return if it's a person
                name = result.get("name", "")
                description = result.get("description", "")
                detailed_description = result.get("detailedDescription", {}).get("articleBody", "")
                wikipedia_url = result.get("detailedDescription", {}).get("url", "")

                summary_parts = []
                if name:
                    summary_parts.append(name)
                if description:
                    summary_parts.append(description)
                if detailed_description:
                    summary_parts.append(detailed_description)
                elif wikipedia_url:
                    summary_parts.append(f"See more at: {wikipedia_url}")

                return " ".join(summary_parts).strip()
    return None

def summarize_differences(kg_text, gpt_text):
    kg_words = set(kg_text.lower().split())
    gpt_words = set(gpt_text.lower().split())
    missing_words = kg_words - gpt_words
    important_missing = [w for w in missing_words if len(w) > 4 and w.isalpha()]
    return important_missing[:10]

def run_kg_comparison(person, api_key):
    report_lines = []

    # Step 1: Knowledge Graph
    kg_data = get_knowledge_graph_data(person, api_key)
    if kg_data:
        report_lines.append(f"=== Knowledge Graph Data for {person} ===")
        report_lines.append(kg_data)
    else:
        report_lines.append(f"No data found in Knowledge Graph for {person}")
        kg_data = ""

    # Step 2: GPT-2 Generation
    prompt = f"Write a short biography of {person}."
    generator = pipeline("text-generation", model="gpt2", device=-1)
    gpt_output = generator(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]
    report_lines.append(f"\n=== GPT-2 Generated Biography ===")
    report_lines.append(gpt_output)

    # Step 3: Semantic Similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_kg = model.encode(kg_data, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_kg).item()
    report_lines.append(f"\nSemantic Similarity to Knowledge Graph Data: {similarity:.2f}")
    if similarity < 0.6:
        report_lines.append("⚠️ GPT-2 output is SUSPICIOUS.")
    else:
        report_lines.append("✅ GPT-2 output looks consistent with Knowledge Graph data.")

    # Step 4: Differences
    report_lines.append("\n=== Summary of Differences (Missing Key Terms) ===")
    if kg_data:
        missing_keywords = summarize_differences(kg_data, gpt_output)
        if missing_keywords:
            for word in missing_keywords:
                report_lines.append(f"❌ MISSING KEYWORD: {word}")
        else:
            report_lines.append("✅ All key terms from Knowledge Graph were referenced.")

    return "\n".join(report_lines)



#comment out if running main script/ not testing specific people
# if __name__ == "__main__":
#     person = "Marie Curie"  # put name of who u wanna test on
#     api_key = "AIzaSyAThq60TW04NeCrA0b_LuAf--DO-g2_mFA"
#     print(run_kg_comparison(person, api_key))

