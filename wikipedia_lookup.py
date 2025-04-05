#wikipedia_lookup.py


import requests

def get_wikipedia_summary(person_name):
    # searches for the url and tries to get exact match
    search_url = f"https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": person_name,
        "format": "json"
    }
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()

    if not search_data["query"]["search"]:
        return f"No Wikipedia page found for '{person_name}'."

    # gets the page title
    page_title = search_data["query"]["search"][0]["title"]

    # how we get summary of the wikipedia page
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    summary_response = requests.get(summary_url)
    if summary_response.status_code == 200:
        summary_data = summary_response.json()
        return summary_data.get("extract", "No summary available.")
    else:
        return f"Failed to fetch summary for '{page_title}'."


if __name__ == "__main__":
    person = "Ada Lovelace"
    summary = get_wikipedia_summary(person)
    print(f"=== Wikipedia Summary for {person} ===")
    print(summary)