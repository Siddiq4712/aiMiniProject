# utils.py
import requests
import json
import pandas as pd
from typing import Tuple, List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"  # default local Ollama generation endpoint

def call_ollama(prompt: str, model: str = "mistral", stream: bool = False, timeout: int = 180) -> Tuple[bool, str]:
    """
    Call local Ollama HTTP API to generate completion from model.
    Returns (success, text_response)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False # Hardcoded as per current app, not using streaming output
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Ollama's /api/generate (non-streaming) returns 'response' key for the generated text
        text = data.get("response")
        if text is None: # Fallback if for some reason 'response' isn't there or format changes
            text = json.dumps(data, ensure_ascii=False) # Safely convert whole data to string
        return True, text
    except Exception as e:
        return False, f"Error calling Ollama: {str(e)}"

def safe_parse_json_array(text: str) -> Tuple[bool, List[Dict] or str]:
    """
    Try to parse JSON array from response text; if the model included extra text,
    attempt to extract the first JSON array found.
    """
    text = text.strip()
    # quick attempt: if it starts with '[' try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return True, parsed
    except Exception:
        pass

    # fallback: find first '[' and last ']' and try to parse
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return True, parsed
        except Exception as e:
            return False, f"Failed to parse JSON candidate: {e}\nCandidate:\n{candidate}"
    return False, "No JSON array could be extracted from model output."

def testcases_to_dataframe(testcases: List[Dict]) -> pd.DataFrame:
    rows = []
    for tc in testcases:
        rows.append({
            "id": tc.get("id"),
            "title": tc.get("title"),
            "preconditions": " | ".join(tc.get("preconditions", [])) if tc.get("preconditions") else "",
            "steps": "\n".join([f"{i+1}. {s}" for i,s in enumerate(tc.get("steps", []))]),
            "expected_results": "\n".join([f"{i+1}. {e}" for i,e in enumerate(tc.get("expected_results", []))]),
            "priority": tc.get("priority"),
            "type": tc.get("type"),
            "acceptance_criteria": " | ".join(tc.get("acceptance_criteria", [])) if tc.get("acceptance_criteria") else "", # Expects list, joins
            "notes": tc.get("notes", "")
        })
    return pd.DataFrame(rows)

def export_testcases_json(testcases: List[Dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(testcases, f, ensure_ascii=False, indent=2)

def export_testcases_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)
