# wikidata_lookup.py

import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_wikidata_id(person_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": person_name,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params).json()
    if response["search"]:
        return response["search"][0]["id"]
    return None

def get_wikidata_facts(entity_id):
    sparql_query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{entity_id} ?prop ?value .
      ?property wikibase:directClaim ?prop .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 20
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": sparql_query}, headers=headers)
    data = response.json()

    facts = {}
    for item in data["results"]["bindings"]:
        prop = item["propertyLabel"]["value"]
        val = item["valueLabel"]["value"]
        facts[prop] = val
    return facts

def run_wikidata_comparison(person, model_name="gpt2"):
    lines = []
    entity_id = get_wikidata_id(person)
    if not entity_id:
        return f"No Wikidata entity found for {person}"

    facts = get_wikidata_facts(entity_id)
    
    fact_text = ". ".join([f"{k}: {v}" for k, v in facts.items()])

    lines.append(f"=== Wikidata Facts for {person} (ID: {entity_id}) ===")
    for prop, val in facts.items():
        lines.append(f"{prop}: {val}")

    prompt = f"Write a short biography of {person}."

    generator = pipeline("text-generation", model=model_name, device=-1)

    gpt_output = generator(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]
    lines.append(f"\n=== GPT-2 Generated Biography ===")
    lines.append(gpt_output)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(fact_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    lines.append(f"\nSemantic Similarity to Wikidata: {similarity:.2f}")
    lines.append("✅ GPT-2 output looks consistent with Wikidata." if similarity >= 0.6 else "⚠️ GPT-2 output is SUSPICIOUS.")

    lines.append("\n=== Fact Presence in GPT-2 Output (Keyword Check) ===")
    for prop, val in facts.items():
        if prop.lower() in gpt_output.lower() or val.lower() in gpt_output.lower():
            lines.append(f"✅ Mentioned: {prop}: {val}")
        else:
            lines.append(f"❌ MISSING: {prop}: {val}")

    lines.append("\n=== Semantic Match per Fact ===")

    for prop, val in facts.items():
        fact = f"{prop}: {val}"
        emb = model.encode(fact, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embedding_gpt, emb).item()
        status = "✅" if score > 0.5 else "❌"
        lines.append(f"{status} {fact} — Similarity: {score:.2f}")

    return "\n".join(lines)
