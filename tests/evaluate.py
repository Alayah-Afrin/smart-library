"""Measure how good the search is. Run from D:/documents/Smart_Library:
    python tests/evaluate.py
Compares three ways of ranking on tests/eval_queries.csv:
    BM25 only (keywords) | Semantic only (meaning) | Hybrid (what the website uses)
Scores: P@5 (share of the top 5 that are right), nDCG@10 (are right books near the top?), MRR (how high is the first right one?)
A result is "right" when its title / author / topic text matches the patterns in the CSV.
Note: topic queries use a keyword proxy for 'right', so treat those numbers as an approximation.
"""
import sys
import csv
import math
import re
import argparse
import warnings
warnings.filterwarnings("ignore", message="This pattern")

sys.path.append('src')
import numpy as np
import pandas as pd
import search as _search
from search import load_search_system, search_books, fold, strip_stopwords

# measure our own catalog only: no internet calls during the evaluation
_search.live_lookup = lambda *a, **k: []
_search.get_better_cover = lambda isbn, title, author, fallback: fallback

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data/processed')
ap.add_argument('--queries', default='tests/eval_queries.csv')
args = ap.parse_args()

system = load_search_system(args.data_dir)
wd = system['work_dedup'].reset_index(drop=True)
wd['blob'] = (wd['title_f'] + ' ' + wd['categories'].fillna('').map(fold) + ' '
              + wd['description'].fillna('').astype(str).str.slice(0, 400).map(fold)).str.replace(r'\s+', ' ', regex=True)
isbn_to_row = {}
for i, isbn in enumerate(wd['isbn13']):
    isbn_to_row.setdefault(isbn, i)


def relevant_mask(row):
    mask = pd.Series(False, index=wd.index)
    for pat, col in ((row['title_pat'], 'title_f'), (row['author_pat'], 'author_f'), (row['topic_pat'], 'blob')):
        if isinstance(pat, str) and pat.strip():
            mask |= wd[col].str.contains(pat, case=False, regex=True, na=False)
    return mask


def dedup_keys(rows_idx):
    seen, out = set(), []
    for i in rows_idx:
        k = wd.at[i, 'dedup_key']
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rank_bm25(q):
    scores = system['bm25'].get_scores(strip_stopwords(fold(q)).split())
    if scores.max() <= 0:
        return []
    return dedup_keys(np.argsort(-scores)[:60])


def rank_semantic(q):
    vec = system['model_embed'].encode([q]).astype('float32')
    dist, idx = system['faiss_index'].search(vec, k=50)
    rows = [isbn_to_row.get(system['isbn13_by_position'][i]) for i in idx[0]]
    return dedup_keys([r for r in rows if r is not None])


def rank_hybrid(q):
    df, _ = search_books(q, system, top_k=24)
    return list(dict.fromkeys(df['dedup_key'].tolist())) if 'dedup_key' in df.columns else \
        dedup_keys([isbn_to_row[i] for i in df['isbn13'] if i in isbn_to_row])


def metrics(keys, rel_keys):
    hits = [1 if k in rel_keys else 0 for k in keys[:10]]
    p5 = sum(hits[:5]) / 5
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel_keys))))
    ndcg = dcg / ideal if ideal else 0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0)
    return p5, ndcg, mrr


MODES = {'BM25': rank_bm25, 'Semantic': rank_semantic, 'Hybrid': rank_hybrid}
rows = []
skipped = []
with open(args.queries, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rel = set(wd.loc[relevant_mask(r), 'dedup_key'])
        if not rel:
            skipped.append(r['query'])
            continue
        for mode, fn in MODES.items():
            keys = fn(r['query'])
            p5, ndcg, mrr = metrics(keys, rel)
            top = [wd.loc[wd['dedup_key'] == k, 'title'].iloc[0] for k in keys[:3]]
            rows.append({'type': r['type'], 'query': r['query'], 'mode': mode, 'P@5': p5, 'nDCG@10': ndcg, 'MRR': mrr,
                         'n_relevant_in_catalog': len(rel), 'top3': ' | '.join(top)})

res = pd.DataFrame(rows)
if res.empty:
    sys.exit("No query had any matching book in the catalog.")
res.to_csv('tests/eval_results.csv', index=False)

print(f"\nQueries scored: {res['query'].nunique()}   skipped (nothing in the catalog matches): {len(skipped)}")
if skipped:
    print("  skipped:", ', '.join(skipped))
cols = ['P@5', 'nDCG@10', 'MRR']
print("\n=== OVERALL (higher is better, max 1.00) ===")
print(res.groupby('mode')[cols].mean().round(2).reindex(list(MODES)).to_string())
print("\n=== BY QUERY TYPE: nDCG@10 ===")
print(res.pivot_table(index='type', columns='mode', values='nDCG@10', aggfunc='mean').round(2)[list(MODES)].to_string())
bad = res[(res['mode'] == 'Hybrid') & (res['nDCG@10'] < 0.3)]
if len(bad):
    print("\n=== HYBRID FAILURES (nDCG@10 below 0.30) ===")
    for _, b in bad.iterrows():
        print(f"  [{b['type']}] {b['query']!r}  ->  {b['top3']}")
print("\nPer-query details saved to tests/eval_results.csv")