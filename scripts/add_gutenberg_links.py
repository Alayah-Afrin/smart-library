"""Match our books to Project Gutenberg (free, legal, public-domain full text).

Run from D:/documents/Smart_Library:
    python scripts/add_gutenberg_links.py
Reads : data/raw/pg_catalog.csv, data/processed/books_search_ready.parquet
Writes: data/processed/gutenberg_links.parquet  (dedup_key, gutenberg_id, read_url)
"""
import sys
import argparse
import pandas as pd

sys.path.append('src')
from search import make_dedup_key

ap = argparse.ArgumentParser()
ap.add_argument('--catalog', default='data/raw/pg_catalog.csv')
ap.add_argument('--data_dir', default='data/processed')
args = ap.parse_args()

cat = pd.read_csv(args.catalog)
print("Gutenberg catalog rows:", len(cat))

# English text books only (skip audio, music, images)
cat = cat[(cat['Type'] == 'Text') & cat['Language'].fillna('').str.split(';').apply(lambda x: 'en' in [s.strip() for s in x])]
cat = cat[cat['Title'].notna() & cat['Authors'].notna()].copy()
print("English text books:", len(cat))


def clean_title(t):
    return str(t).split('\n')[0].strip()          # some titles hold a second line (subtitle)


def clean_author(a):
    first = str(a).split(';')[0].strip()          # first listed author
    parts = [p.strip() for p in first.split(',')]
    parts = [p for p in parts if p and not any(ch.isdigit() for ch in p)]   # drop "1775-1817"
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}"           # "Austen, Jane" -> "Jane Austen"
    return parts[0] if parts else ''


cat['dedup_key'] = [make_dedup_key(clean_title(t), clean_author(a)) for t, a in zip(cat['Title'], cat['Authors'])]
cat = cat.sort_values('Text#').drop_duplicates('dedup_key')          # keep the lowest id per work

books = pd.read_parquet(f'{args.data_dir}/books_search_ready.parquet', columns=['title', 'author'])
books['dedup_key'] = [make_dedup_key(t, a) for t, a in zip(books['title'], books['author'].fillna(''))]

links = cat[cat['dedup_key'].isin(set(books['dedup_key']))][['dedup_key', 'Text#']].rename(columns={'Text#': 'gutenberg_id'})
links['read_url'] = 'https://www.gutenberg.org/ebooks/' + links['gutenberg_id'].astype(str)
links.to_parquet(f'{args.data_dir}/gutenberg_links.parquet', index=False)

matched = books['dedup_key'].isin(set(links['dedup_key'])).sum()
print(f"Matched {len(links)} Gutenberg books -> {matched} of our {len(books)} books get a 'Read free' link")
print(links.head(8).to_string(index=False))