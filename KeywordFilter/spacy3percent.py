#!/usr/bin/env python3
"""
HARSH vocabulary filter → harsh.json
• ASCII letters only, length 4–12
• Drop profanity + stop-words
• Zipf 3.0–4.5  (wordfreq)
• In top-40 k English lemmas
• POS must be NOUN or PROPN (spaCy)
"""

import json, re, sys, random, warnings, functools
from pathlib import Path
from wordfreq import zipf_frequency, top_n_list
from better_profanity import profanity

# ─── spaCy (POS) ──────────────────────────────────────────────────────
import spacy, nltk
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # fast tagger only

from nltk.corpus import stopwords

# ─── resources ────────────────────────────────────────────────────────
stop     = set(stopwords.words("english"))
lexicon  = set(top_n_list("en", 40_000))
rx       = re.compile(r"^[A-Za-z]{4,12}$")      # stricter word shape

@functools.lru_cache(maxsize=10_000)
def pos_is_noun(word: str) -> bool:
    """spaCy POS test (cached)"""
    doc = nlp(word)
    return bool(doc) and doc[0].pos_ in {"NOUN", "PROPN"}

# ─── predicate ────────────────────────────────────────────────────────
def ok(w: str) -> bool:
    w = w.lower()
    if not rx.fullmatch(w): return False
    if w in stop or profanity.contains_profanity(w): return False
    f = zipf_frequency(w, "en")
    if f < 3.0 or f > 4.5: return False
    if w not in lexicon: return False
    if not pos_is_noun(w): return False
    return True

# ─── main ─────────────────────────────────────────────────────────────
def run(src, dst="harsh.json", frac=0.05, seed=42):
    words = json.loads(Path(src).read_text())
    keep  = [w for w in words if ok(w)]

    # optional down-sample to an exact fraction (default 5 %)
    target = int(frac * len(words))
    if len(keep) > target:
        random.seed(seed)
        keep = random.sample(keep, target)

    Path(dst).write_text(json.dumps(sorted(set(keep))))
    print(f"kept {len(keep)} / {len(words)} ({len(keep)/len(words):.2%}) → {dst}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python harsh_filter.py <filtered.json> [harsh.json]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "harsh.json")
