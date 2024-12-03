import requests
import urllib.parse
from bs4 import BeautifulSoup
import json

def fetch_crossword_answers(clue: str, pattern: str = None, length: int = None):
    base_url = "https://www.dictionary.com/e/crosswordsolver/"
    url = base_url + f"{urllib.parse.quote(clue)}/"

    queries = []
    if pattern and length:
        queries.append({"p": pattern, "l": length})
    if pattern:
        queries.append({"p": pattern})
    queries.append({})

    for params in queries:
        try:
            request_url = url + f"?{urllib.parse.urlencode(params)}" if params else url
            print(f"Fetching: {request_url}", file=sys.stderr)  # Debug: Log the request URL
            response = requests.get(request_url)
            response.raise_for_status()

            print(f"Response status: {response.status_code}", file=sys.stderr)  # Debug: Status code
            print(f"Raw response: {response.text[:500]}", file=sys.stderr)  # Debug: Log partial response

            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('div', class_='solver-table__row')

            answers = []
            for row in rows:
                answer_cell = row.find('div', attrs={'data-cy': 'result'})
                if answer_cell:
                    answer = answer_cell.text.strip()
                    answers.append(answer)

            if answers:
                return {"answers": answers}  # Return results as a JSON-compatible dict

        except requests.RequestException as e:
            print(f"Error fetching crossword answers for params {params}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)

    return {"answers": []}  # Always return JSON-compatible output


if __name__ == "__main__":
    import sys
    try:
        clue = sys.argv[1]
        pattern = sys.argv[2] if len(sys.argv) > 2 else None
        length = int(sys.argv[3]) if len(sys.argv) > 3 else None

        result = fetch_crossword_answers(clue, pattern, length)
        print(json.dumps(result))  # Ensure the output is always valid JSON
    except Exception as e:
        print(json.dumps({"error": str(e), "answers": []}), file=sys.stderr)
