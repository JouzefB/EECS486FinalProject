import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    LIMIT 50
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

def run_wikidata_comparison(person, model_name="gpt2", gpt_output=None, repeated_prompt=False):
    lines = []
    entity_id = get_wikidata_id(person)
    if not entity_id:
        return {
            "text": f"No Wikidata entity found for {person}",
            "verdict": "⚠️ No entity found",
            "similarity": 0.0,
            "facts_matched": 0,
            "facts_total": 0,
            "missing_keywords": []
        }

    facts = get_wikidata_facts(entity_id)
    fact_text = ". ".join([f"{k}: {v}" for k, v in facts.items()])

    if gpt_output is None:
        prompt = f"Write a short biography of {person}."
        generator = pipeline("text-generation", model=model_name, device=-1)
        gpt_output = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]
        repeated_prompt = gpt_output.lower().count(prompt.lower()) > 2

    lines.append(f"=== Wikidata Facts for {person} (ID: {entity_id}) ===")
    for prop, val in facts.items():
        lines.append(f"{prop}: {val}")

    lines.append(f"\n=== GPT-2 Generated Biography ===")
    lines.append(gpt_output)

    if repeated_prompt:
        lines.append("\n⚠️ Detected prompt repetition — GPT may have echoed the instruction.")

    # === Semantic Similarity ===
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(fact_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    lines.append(f"\nSemantic Similarity Score: {similarity:.2f}")
    final_verdict = "✅ GPT-2 output looks consistent with Wikidata." if similarity >= 0.6 else "⚠️ GPT-2 output is SUSPICIOUS."
    lines.append(final_verdict)

    # === Keyword Match ===
    lines.append("\n=== Fact Presence in GPT-2 Output (Keyword Check) ===")
    missing_keywords = []
    for prop, val in facts.items():
        if prop.lower() in gpt_output.lower() or val.lower() in gpt_output.lower():
            lines.append(f"✅ Mentioned: {prop}: {val}")
        else:
            lines.append(f"❌ MISSING: {prop}: {val}")
            missing_keywords.append(f"{prop}: {val}")

    # === Per-Fact Semantic Match ===
    lines.append("\n=== Semantic Match per Fact ===")
    matched = 0
    for prop, val in facts.items():
        fact = f"{prop}: {val}"
        emb = model.encode(fact, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embedding_gpt, emb).item()
        status = "✅" if score > 0.5 else "❌"
        if score > 0.5:
            matched += 1
        lines.append(f"{status} {fact} — Similarity: {score:.2f}")

    # === Final Summary ===
    total_facts = len(facts)
    summary_line = "✅ Overall Verdict: GPT-2 Output Seems Factual." if similarity >= 0.6 and matched >= total_facts * 0.5 else "⚠️ Overall Verdict: GPT-2 Output is Likely Hallucinated."
    lines.append(f"\n=== Final Summary ===")
    lines.append(f"Semantic Similarity: {similarity:.2f}")
    lines.append(f"Prompt Repeated: {'Yes' if repeated_prompt else 'No'}")
    lines.append(f"Facts Semantically Matched: {matched}/{total_facts}")
    lines.append(summary_line)

    return {
        "text": "\n".join(lines),
        "verdict": summary_line,
        "similarity": similarity,
        "facts_matched": matched,
        "facts_total": total_facts,
        "missing_keywords": missing_keywords[:10]
    }

# CLI for testing
if __name__ == "__main__":
    import argparse
    from transformers import pipeline

    parser = argparse.ArgumentParser(description="Run Wikidata comparison for a given person")
    parser.add_argument("--person", type=str, required=True, help="Name of the person to check")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--tag", type=str, default="sampling", help="sampling or greedy")
    args = parser.parse_args()

    generator = pipeline("text-generation", model=args.model, device=-1)
    prompt = f"Write a short biography of {args.person}."
    gpt_output = generator(prompt, max_length=300, do_sample=(args.tag == "sampling"))[0]["generated_text"]
    repeated_prompt = gpt_output.lower().count(prompt.lower()) > 2

    print(f"🔍 Comparing GPT output to Wikidata for: {args.person}\n")
    result = run_wikidata_comparison(args.person, args.model, gpt_output, repeated_prompt)
    print(result["text"])
