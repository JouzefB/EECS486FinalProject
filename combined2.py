import os
import argparse
import time
import csv
from transformers import pipeline
from compare_gpt2_to_wiki import run_wiki_comparison
from wikidata_lookup import run_wikidata_comparison
from Google_Knowledge_API_Graph_vs_chatgpt2 import run_kg_comparison

##must change google_api_key if it expires, explained in readme how to do so
GOOGLE_API_KEY = "AIzaSyAThq60TW04NeCrA0b_LuAf--DO-g2_mFA"

#list of people being tested on
##can change whoever it is and the amount
PEOPLE = [
    "Marie Curie", "Søren Bjerg", "Albert Einstein", "Timothée Chalamet",
    "Barack Obama", "Serena Williams", "Lionel Messi", "Billie Eilish",
    "Cleopatra", "Rosalind Franklin", "Nikola Tesla", "Elon Musk",
    "Anne Frank", "Taylor Swift", "Emily King"
]

## function to detect common repetition found in GPT-2 output

def detect_prompt_repetition(prompt, output):
    return output.lower().count(prompt.lower()) >= 2

## to help with outputs for the text files and make it look good

def section_header(title):
    return f"{'=' * 10} {title} {'=' * 10}"

def main(model_name, tag, temperature):
    ##CHANGE OUTPUT DIR TO WHATEVER OUTPUT U WANT IT TO BE

    ## maybe shouldve made this an argument people shouldve put in their terminal
    ## i learned infintie lowkey
    output_dir = "real_final_dataset_reports"

    ## but makes directory if doesn't exist which user might not want
    os.makedirs(output_dir, exist_ok=True)

    ## summary csv is to store all the results that seemed significant so we could compare them in a graph or chart
    summary_csv = os.path.join(output_dir, "summary.csv")
    write_header = not os.path.exists(summary_csv)

    with open(summary_csv, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow([
                "Person", "Model", "Tag",
                "Wiki Similarity", "Wiki Verdict", "Wiki Missing Keywords",
                "Wikidata Similarity", "Wikidata Verdict", "Facts Matched", "Facts Total",
                "KG Similarity", "KG Verdict", "KG Missing Keywords",
                "Prompt Repeated", "Run Time (s)"
            ])


        ## loads in the GPT model that user wants
        ## or any model for future researchers 

        generator = pipeline("text-generation", model=model_name, device=-1)

        for person in PEOPLE:
            print(f"\n🔍 Processing {person} with model '{model_name}' and tag '{tag}'...")
            start = time.time()

            ## can change prompt to be wahtever the user wants to input in the LLM model

            prompt = f"Write a short and informative biography of {person}, including important facts such as date of birth, occupation, major achievements, awards, and contributions."
            
            sampling = (tag == "sampling")
            gpt_output = generator(prompt, max_length=300, do_sample=sampling, temperature=temperature)[0]["generated_text"]
            prompt_repeated = detect_prompt_repetition(prompt, gpt_output)

            try:
                ##runs each program w/ the file and then they each run their own process and give back the hallucination verdict
                wiki = run_wiki_comparison(person, model_name, gpt_output, prompt_repeated)
                wikidata = run_wikidata_comparison(person, model_name, gpt_output, prompt_repeated)
                kg = run_kg_comparison(person, GOOGLE_API_KEY, model_name, gpt_output, prompt_repeated)
            except Exception as e:
                print(f"❌ Error processing {person}: {e}")
                continue

            filename = f"{person.replace(' ', '_')}_{model_name.replace('/', '_')}_{tag}.txt"
            filepath = os.path.join(output_dir, filename)

            ##writes results to file

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("========== LEGEND ==========\n")
                f.write("✅ = Output is likely factual or consistent\n")
                f.write("⚠️ = Output is likely hallucinated or contains omissions\n")
                f.write("Semantic Similarity thresholds:\n")
                f.write("  - ✅ ≥ 0.60 → semantically aligned\n")
                f.write("  - ⚠️ < 0.60 → possibly unrelated or hallucinated\n")
                f.write("Prompt Repetition detected if the model repeats the prompt multiple times.\n")
                f.write("Keyword matches are used to identify missing facts or details.\n\n")

                f.write(section_header("Wikipedia Comparison") + "\n")
                f.write(wiki["text"] + "\n\n")

                f.write(section_header("Wikidata Comparison") + "\n")
                f.write(wikidata["text"] + "\n\n")

                f.write(section_header("Google Knowledge Graph Comparison") + "\n")
                f.write(kg["text"] + "\n\n")

                f.write(section_header("Overall Verdicts") + "\n")
                f.write(f"Wikipedia: {wiki['verdict']} (Missing: {wiki['num_missing_keywords']})\n")
                f.write(f"Wikidata: {wikidata['verdict']} ({wikidata['facts_matched']}/{wikidata['facts_total']} facts matched)\n")
                f.write(f"Google KG: {kg['verdict']} (Missing: {kg['num_missing_keywords']})\n")


            ##runtime for each run so user can track that as well

            runtime = round(time.time() - start, 2)
            writer.writerow([
                person, model_name, tag,
                wiki["similarity"], wiki["verdict"], wiki["num_missing_keywords"],
                wikidata["similarity"], wikidata["verdict"], wikidata["facts_matched"], wikidata["facts_total"],
                kg["similarity"], kg["verdict"], kg["num_missing_keywords"],
                "Yes" if prompt_repeated else "No",
                runtime
            ])

            print("📄 Verdicts:")
            print("  Wikipedia:", wiki["verdict"])
            print("  Wikidata:", wikidata["verdict"])
            print("  Google KG:", kg["verdict"])
            print(f"✅ Report saved to: {filename} ({runtime}s)")

    print("\n✅✅ All comparisons completed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination comparison across models")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name (e.g., gpt2, gpt2-medium, gpt2-xl)")
    parser.add_argument("--tag", type=str, default="sampling", help="Tag to label generation strategy (e.g., sampling, greedy)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (only used if tag is 'sampling')")
    args = parser.parse_args()
    main(args.model, args.tag, args.temperature)
