#!/usr/bin/env python3
"""
Ultra-harsh vocabulary filter
• ASCII letters/’/-, length 3–15
• Removes profanity + stop-words
• Keeps only mid-tail Zipf 2.7–5.0
• Must appear in 60 k-word English lexicon
• Optionally subsamples to 10 % of source size
"""

import json, re, sys, random, warnings
from pathlib import Path
from wordfreq import zipf_frequency, top_n_list
from better_profanity import profanity
import nltk
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# ─── resources ─────────────────────────────────────────────────────────
stop     = set(stopwords.words("english"))
lexicon  = set(top_n_list("en", 60_000))      # “real” English lemmas
rx       = re.compile(r"[A-Za-z'-]{3,15}$")   # ASCII word-shape regex

# ─── token predicate ───────────────────────────────────────────────────
def ok(w):
    w = w.lower()
    if not rx.fullmatch(w): return False
    if w in stop: return False
    if profanity.contains_profanity(w): return False
    f = zipf_frequency(w, "en")
    if f < 2.7 or f > 5.0: return False
    if w not in lexicon: return False
    return True

# ─── main ──────────────────────────────────────────────────────────────
def run(src, dst, frac=0.10, seed=42):
    words = json.loads(Path(src).read_text())
    keep  = [w for w in words if ok(w)]

    target = int(frac * len(words))
    if len(keep) > target:
        random.seed(seed)
        keep = random.sample(keep, target)

    Path(dst).write_text(json.dumps(sorted(set(keep))))
    print(f"kept {len(keep)} / {len(words)} ({len(keep)/len(words):.2%})")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
