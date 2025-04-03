# combined2.py
from compare_gpt2_to_wiki import run_wiki_comparison
from wikidata_lookup import run_wikidata_comparison
from Google_Knowledge_API_Graph_vs_chatgpt2 import run_kg_comparison

import os

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

os.makedirs("hallucination_reports", exist_ok=True)

for person in PEOPLE:
    print(f"Processing {person}...")
    wiki_result = run_wiki_comparison(person)
    wikidata_result = run_wikidata_comparison(person)
    kg_result = run_kg_comparison(person, GOOGLE_API_KEY)

    with open(f"hallucination_reports/{person.replace(' ', '_')}.txt", "w") as f:
        f.write("=== Wikipedia Comparison ===\n")
        f.write(wiki_result + "\n\n")
        f.write("=== Wikidata Comparison ===\n")
        f.write(wikidata_result + "\n\n")
        f.write("=== Google Knowledge Graph Comparison ===\n")
        f.write(kg_result + "\n")

print("✅ All reports written.")
