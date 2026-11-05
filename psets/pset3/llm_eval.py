"""Evaluate an LLM with GroqCloud; repeatable prompts and JSONL caching.

Set GROQ_API_KEY in your environment. Use --dry-run to preview without calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import pandas as pd
from data_utils import EXCLUDE_FROM_FEATURES

DEFAULT_MODEL = "openai/gpt-oss-20b"
PRIVATE_COLUMNS = EXCLUDE_FROM_FEATURES | {"target", "age_group"}

RISK_FIRST =  ["discharge_disposition_id", "time_in_hospital", "num_medications",
    "number_diagnoses", "age"]

def make_prompt(row: dict, format_name: str) -> str:
    """Build two example representations; students design the third."""
    if format_name not in {'structured', 'natural', 'alternative'}:
        raise ValueError('Unknown format')
    #if format_name == 'alternative':
        #raise NotImplementedError('Student TODO: design and justify a third prompt format')


    # Never provide the label, derived outcomes, or patient/encounter identifiers.
    fields = []

    for name, value in row.items():
        if name in PRIVATE_COLUMNS or name.startswith('Unnamed:'):
            continue
        if pd.isna(value):
            value = 'missing'
        fields.append((name, str(value)))
    if format_name == 'alternative':
        rank = {n: i for i, n in enumerate(RISK_FIRST)}
        fields = sorted(fields, key=lambda kv: rank.get(kv[0], len(RISK_FIRST)))
    

    instruction = (
        'Estimate the probability that this patient will be readmitted to a '
        'hospital within 30 days after discharge. Return only a single number '
        'between 0 and 1 (inclusive), with no explanation.\n\n'
    )
    if format_name == 'structured':
        representation = 'Patient encounter features (key: value):\n' + '\n'.join(
            f'{name}: {value}' for name, value in fields
        )
    else:
        representation = (
            'Patient encounter description: The recorded attributes are '
            + '; '.join(f'{name.replace("_", " ")} is {value}' for name, value in fields)
            + '.'
        )
    return instruction + representation


def parse_probability(text: str):
    """Strictly parse one probability; invalid model responses become NaN."""
    if not re.fullmatch(r"\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*", text):
        return float("nan")
    return float(text.strip())


def cached_call(client, model, prompt, cache_path, max_retries=3):
    key = hashlib.sha256((model + "\n" + prompt).encode()).hexdigest()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                item = json.loads(line)
                if item["key"] == key:
                    return item["response"]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
            model=model, temperature=0, max_tokens=1024,
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}])
            response = completion.choices[0].message.content or ""
            with cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "model": model,
                                    "response": response}) + "\n")
            return response
        except Exception as exc:
            # For real deployments, inspect status codes/retry-after rather than
            # retrying all errors. Never print or log your API key.
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="outputs/llm_sample.csv")
    ap.add_argument("--format", choices=["structured", "natural", "alternative"], default="structured")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process first N sampled encounters")
    args = ap.parse_args()
    frame = pd.read_csv(args.sample)
    if args.limit is not None:
        frame = frame.head(args.limit)
    client = None
    if not args.dry_run:
        from groq import Groq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set; use --dry-run to preview")
        client = Groq()
    output = Path("outputs") / f"llm_{args.format}.csv"
    rows = []
    for i, (_, record) in enumerate(frame.iterrows()):
        prompt = make_prompt(record.to_dict(), args.format)
        if args.dry_run:
            if i == 0:
                print(prompt)
            continue
        reply = cached_call(client, args.model, prompt, "outputs/llm_cache.jsonl")
        rows.append({"sample_row": i, "target": record["target"],
                     "age_group": record.get("age_group", ""),
                     "response": reply, "probability": parse_probability(reply)})
        # Save progress even if later requests fail.
        pd.DataFrame(rows).to_csv(output, index=False)
    if args.dry_run:
        print(f"Dry run completed for {len(frame)} encounters; no API calls made")
    else:
        print(f"Saved {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
