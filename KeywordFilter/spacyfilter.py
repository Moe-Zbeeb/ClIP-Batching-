#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
from wordfreq import zipf_frequency
from better_profanity import profanity
from spacy.lang.en.stop_words import STOP_WORDS
from concurrent.futures import ThreadPoolExecutor as TPE

rx = re.compile(r"[a-zA-Z'-]+$")

def ok(w):
    w = w.lower()
    if not rx.fullmatch(w): return False
    if w in STOP_WORDS: return False
    if profanity.contains_profanity(w): return False
    if zipf_frequency(w, "en") == 0: return False
    return True

def run(src, dst, workers=8):
    words = json.loads(Path(src).read_text())
    with TPE(max_workers=workers) as ex:
        keep = list(filter(None, ex.map(lambda x: x if ok(x) else None, words)))
    Path(dst).write_text(json.dumps(sorted(set(keep))))
    print(f"kept {len(keep)} / {len(words)}")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
