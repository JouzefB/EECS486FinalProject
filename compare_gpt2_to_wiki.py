# compare_gpt2_to_wiki.py

import os
from sentence_transformers import SentenceTransformer, util
from wikipedia_lookup import get_wikipedia_summary
import re

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def extract_keywords(text):
    words = re.findall(r"\b\w{5,}\b", text.lower())
    return set(words)

def run_wiki_comparison(person, model_name="gpt2", gpt_output=None, repeated_prompt=False):
    lines = []
    header = lambda title: f"\n{'=' * 10} {title} {'=' * 10}"

    prompt = f"Write a short biography of {person}."

    if gpt_output is None:
        raise ValueError("GPT output must be provided externally.")

    # Header and GPT output
    lines.append(header(f"GPT-2 Generated Biography for {person}"))
    lines.append(gpt_output)
    if repeated_prompt:
        lines.append("\n⚠️ Detected prompt repetition — GPT may have echoed the instruction.")

    # Wikipedia reference
    wiki_summary = get_wikipedia_summary(person)
    lines.append(header("Wikipedia Summary"))
    lines.append(wiki_summary)

    # Semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_gpt = model.encode(gpt_output, convert_to_tensor=True)
    embedding_wiki = model.encode(wiki_summary, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_gpt, embedding_wiki).item()

    lines.append(header("Semantic Similarity Score"))
    lines.append(f"Cosine Similarity Score: {similarity:.2f}")
    final_verdict = "✅ GPT-2 output looks consistent with Wikipedia." if similarity >= 0.6 else "⚠️ GPT-2 output is potentially hallucinated or unrelated."
    lines.append(final_verdict)

    # Keyword comparison
    wiki_keywords = extract_keywords(wiki_summary)
    gpt_keywords = extract_keywords(gpt_output)
    missing_keywords = sorted(wiki_keywords - gpt_keywords)

    lines.append(header("Top Missing Keywords"))
    if missing_keywords:
        for word in missing_keywords[:10]:
            lines.append(f"❌ MISSING: {word}")
    else:
        lines.append("✅ All major Wikipedia keywords are present.")

    # Sentence-level matching
    gpt_sents = [s for s in gpt_output.split(".") if len(s.split()) > 5]
    wiki_sents = [s for s in wiki_summary.split(".") if len(s.split()) > 5]
    max_score = 0.0
    best_pair = ("", "")
    for g in gpt_sents:
        for w in wiki_sents:
            score = util.pytorch_cos_sim(model.encode(g, convert_to_tensor=True), model.encode(w, convert_to_tensor=True)).item()
            if score > max_score:
                max_score = score
                best_pair = (w.strip(), g.strip())

    lines.append(header("Closest Matching Sentence Pair"))
    if best_pair[0] and best_pair[1]:
        lines.append(f"[Wikipedia] {best_pair[0]}")
        lines.append(f"[GPT-2]     {best_pair[1]}")
        lines.append(f"→ Cosine Similarity: {max_score:.2f}")
    else:
        lines.append("No meaningful sentence match found.")

    # Final summary
    lines.append(header("Verdict Summary"))
    lines.append(f"Semantic Similarity: {similarity:.2f}")
    lines.append(f"Missing Keywords: {len(missing_keywords)}")
    lines.append("✅ Biography is semantically aligned with Wikipedia." if similarity >= 0.6 else "⚠️ Possible hallucination or deviation from trusted summary.")

    return {
        "text": "\n".join(lines),
        "verdict": final_verdict,
        "similarity": similarity,
        "num_missing_keywords": len(missing_keywords),
        "missing_keywords": missing_keywords[:10]  # top 10 only
    }

if __name__ == "__main__":
    import argparse
    from transformers import pipeline

    parser = argparse.ArgumentParser(description="Compare GPT-2 biography with Wikipedia summary.")
    parser.add_argument("--person", required=True, help="Name of the person to query")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model name (default: gpt2)")
    parser.add_argument("--tag", default="sampling", help="Tag to determine generation strategy")
    args = parser.parse_args()

    prompt = f"Write a short biography of {args.person}."
    generator = pipeline("text-generation", model=args.model, device=-1)
    gpt_output = generator(prompt, max_length=300, do_sample=(args.tag == "sampling"))[0]["generated_text"]

    repeated_prompt = gpt_output.lower().count(prompt.lower()) > 2
    print(f"\n🔍 Comparing GPT-2 output to Wikipedia for: {args.person}\n")
    result = run_wiki_comparison(args.person, model_name=args.model, gpt_output=gpt_output, repeated_prompt=repeated_prompt)
    print(result["text"])
