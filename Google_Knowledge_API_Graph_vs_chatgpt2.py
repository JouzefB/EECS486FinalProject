import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##function to get knowledge graph data from Google's Knowledge Graph API
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

## for the outputs to summarize the differences between knowledge graph and gpt output
def summarize_differences(kg_text, gpt_text):
    kg_words = set(kg_text.lower().split())
    gpt_words = set(gpt_text.lower().split())
    missing_words = kg_words - gpt_words
    return [w for w in missing_words if len(w) > 4 and w.isalpha()][:10]


## main function that compares the output of the Google API facts vs the LLM model
def run_kg_comparison(person, api_key, model_name="gpt2", gpt_output=None, repeated_prompt=False):
    lines = []
    header = lambda title: f"\n{'=' * 10} {title} {'=' * 10}"

    # Get Knowledge Graph summary
    kg_text = get_knowledge_graph_data(person, api_key)
    lines.append(header(f"Knowledge Graph Data for {person}"))
    if kg_text:
        lines.append(kg_text)
    else:
        lines.append("[❌] No data found in Knowledge Graph.")
        return {
            "text": "\n".join(lines),
            "verdict": "⚠️ No Knowledge Graph data found.",
            "similarity": 0.0,
            "num_missing_keywords": 0,
            "missing_keywords": []
        }

    # Generate GPT-2 output only if not passed
    if gpt_output is None:
        generator = pipeline("text-generation", model=model_name, device=-1)
        prompt = f"Write a short biography of {person}."
        gpt_output = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]
        repeated_prompt = gpt_output.lower().count(prompt.lower()) > 2

    if repeated_prompt:
        lines.append("⚠️ Detected prompt repetition — GPT may have echoed the instruction.")

    lines.append(header("GPT-2 Generated Biography"))
    lines.append(gpt_output)

    # Semantic Similarity Calculation
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)

    embedding_kg = model.encode(kg_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_kg).item()

    lines.append(header("Semantic Similarity Score"))
    lines.append(f"Cosine Similarity Score: {similarity:.2f}")

    semantic_ok = similarity >= 0.6
    lines.append("✅ GPT-2 output aligns semantically with Knowledge Graph." if semantic_ok else "⚠️ GPT-2 output is semantically inconsistent or unrelated.")

    # Keyword Difference Summary
    lines.append(header("Missing Key Terms from GPT-2 Output"))

    missing_keywords = summarize_differences(kg_text, gpt_output)
    
    for word in missing_keywords:
        lines.append(f"❌ MISSING: {word}")
    if not missing_keywords:
        lines.append("✅ All major keywords present in GPT-2 output.")

    #fincal verdict summary printing
    lines.append(header("Verdict Summary"))
    lines.append(f"Semantic Score: {similarity:.2f}")
    lines.append(f"Missing Keywords: {len(missing_keywords)}")
    lines.append(f"Prompt Repeated: {'Yes' if repeated_prompt else 'No'}")

    if semantic_ok and len(missing_keywords) <= 3:
        verdict = "✅ GPT-2 output PASSES semantic + factual check."
    else:
        verdict = "⚠️ GPT-2 output LIKELY contains hallucinations or omissions."
    lines.append(verdict)

    return {
        "text": "\n".join(lines),
        "verdict": verdict,
        "similarity": similarity,
        "num_missing_keywords": len(missing_keywords),
        "missing_keywords": missing_keywords
    }

# if want to test by itself and own model instead of the combined FACT or AI? pipeline
if __name__ == "__main__":
    import argparse
    from transformers import pipeline

    parser = argparse.ArgumentParser(description="Compare GPT-2 biography with Google Knowledge Graph data.")
    parser.add_argument("--person", required=True, help="Name of the person to query")
    parser.add_argument("--api_key", required=True, help="Google Knowledge Graph API key")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model name (default: gpt2)")
    parser.add_argument("--tag", default="sampling", help="sampling or greedy")
    args = parser.parse_args()

    prompt = f"Write a short biography of {args.person}."
    generator = pipeline("text-generation", model=args.model, device=-1)
    gpt_output = generator(prompt, max_length=300, do_sample=(args.tag == "sampling"))[0]["generated_text"]
    repeated = gpt_output.lower().count(prompt.lower()) > 2

    result = run_kg_comparison(args.person, args.api_key, args.model, gpt_output, repeated)
    print(result["text"])
