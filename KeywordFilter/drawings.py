#!/usr/bin/env python3
import json, sys, nltk, networkx as nx
from pathlib import Path
from nltk.corpus import wordnet as wn

# one-time downloads (silent if already present)
nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)

words = json.loads(Path(sys.argv[1]).read_text())
G = nx.DiGraph()

def add_path(word):
    ss = wn.synsets(word, pos="n")
    if not ss: return
    path = min(ss[0].hypernym_paths(), key=len)        # shortest chain
    names = [s.name().split('.')[0] for s in path]     # lemma only

    # e.g. ['entity', 'animal', 'mammal', 'feline', 'cat']
    for parent, child in zip(names, names[1:]):
        G.add_edge(parent, child)
    # finally map the word itself as a leaf alias of its synset name
    if word != names[-1]:
        G.add_edge(names[-1], word)

for w in words:
    add_path(w)

print(f"graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
nx.nx_pydot.write_dot(G, "hierarchy.dot")
print("Wrote hierarchy.dot → run:  dot -Tpng hierarchy.dot -o tree.png")
