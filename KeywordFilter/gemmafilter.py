#!/usr/bin/env python3
"""
Gemma-only English-word filter.
  • Reads a JSON list of tokens.
  • Feeds Gemma batches of 100 tokens.
  • Keeps tokens Gemma answers “YES”.
  • Writes <src>/filtered.json unless -o/--dst is given.

Requires:  requests, tqdm   (pip install requests tqdm)
Assumes:   Ollama endpoint  http://localhost:11434/api/generate
           with Gemma pulled  (ollama pull gemma3:4b)
           and Ollama set to allow --workers concurrent requests:
             ~/.ollama/config.yaml
               concurrency: 3         # ← same as --workers
"""

import argparse, json, requests, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

YES_RE = re.compile(r"^y(es)?\.?$", re.I)


def ask_batch(model: str, host: str, batch: list[str], timeout: int = 60) -> list[bool]:
    """Return YES/NO flags for every token in *batch* (same order)."""
    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(batch))
    prompt = (
        "For EACH token answer only YES or NO, one per line, SAME order.\n"
        "Say YES ONLY if the token is a STANDARD English dictionary word OR a well-known proper noun "
        "(e.g. 'London', 'Microsoft'). Reject abbreviations, slang, jargon, non-English, misspellings, "
        "nonsense, or anything offensive.\n\n"
        "Tokens:\n" + numbered
    )

    for _ in range(3):  # retry up to 3×
        try:
            r = requests.post(
                f"{host}/api/generate",
                json={"model": model,
                      "prompt": prompt,
                      "stream": False},
                timeout=timeout,
            )
            if r.status_code == 200:
                lines = r.json()["response"].strip().splitlines()
                return [
                    i < len(lines) and YES_RE.match(lines[i]) is not None
                    for i in range(len(batch))
                ]
        except requests.exceptions.RequestException:
            time.sleep(1)
    # on repeated failure mark all NO
    return [False] * len(batch)


def chunks(lst: list[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="input JSON (list of tokens)")
    p.add_argument("-o", "--dst", help="output JSON (default <src>/filtered.json)")
    p.add_argument("--model",   default="gemma3:4b")
    p.add_argument("--host",    default="http://localhost:11434")
    p.add_argument("--batch",   type=int, default=100, help="tokens per request")
    p.add_argument("--workers", type=int, default=3,   help="parallel batch calls")
    args = p.parse_args()

    # ─── read input ─────────────────────────────────────────────────────
    try:
        words = json.load(open(args.src, encoding="utf-8"))
    except Exception:
        sys.exit("cannot read input JSON")
    if not isinstance(words, list):
        sys.exit("input JSON must be a list")
    words = [w for w in words if isinstance(w, str)]
    if not words:
        sys.exit("no valid strings found in input")

    dst = Path(args.dst) if args.dst else Path(args.src).with_name("filtered.json")

    # ─── submit batches ─────────────────────────────────────────────────
    kept: list[str] = []
    batches = list(chunks(words, args.batch))
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
         tqdm(total=len(words), unit="tok", ncols=80) as bar:

        futures = {pool.submit(ask_batch, args.model, args.host, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            flags = fut.result()
            kept.extend(w for w, ok in zip(batch, flags) if ok)
            bar.update(len(batch))

    # ─── write output ───────────────────────────────────────────────────
    json.dump(kept, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{len(kept)}/{len(words)} kept  →  {dst}")


if __name__ == "__main__":
    main()
