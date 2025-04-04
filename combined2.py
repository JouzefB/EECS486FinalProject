# modified_combined.py
import os
import argparse
from compare_gpt2_to_wiki import run_wiki_comparison
from wikidata_lookup import run_wikidata_comparison
from Google_Knowledge_API_Graph_vs_chatgpt2 import run_kg_comparison

GOOGLE_API_KEY = "AIzaSyAThq60TW04NeCrA0b_LuAf--DO-g2_mFA"
PEOPLE = [
    "Marie Curie",
    "Søren Bjerg",
    "Albert Einstein",
    "Timothée Chalamet",
    "Barack Obama",
    "Serena Williams",
    "Lionel Messi",
    "Billie Eilish",
    "Cleopatra",
    "Rosalind Franklin"
]

def main(model_name):
    os.makedirs("hallucination_reports", exist_ok=True)

    for person in PEOPLE:
        print(f"Processing {person} with model {model_name}...")
        wiki_result = run_wiki_comparison(person, model_name)
        wikidata_result = run_wikidata_comparison(person, model_name)
        kg_result = run_kg_comparison(person, GOOGLE_API_KEY, model_name)

        filename = f"{person.replace(' ', '_')}_{model_name.replace('/', '_')}_do_sample_true_temp_0_7.txt"
        filepath = os.path.join("hallucination_reports", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Wikipedia Comparison ===\n")
            f.write(wiki_result + "\n\n")
            f.write("=== Wikidata Comparison ===\n")
            f.write(wikidata_result + "\n\n")
            f.write("=== Google Knowledge Graph Comparison ===\n")
            f.write(kg_result + "\n")

    print("\n✅ All reports written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination comparison across models")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name (e.g., gpt2, gpt2-medium, gpt2-xl)")
    args = parser.parse_args()
    main(args.model)
