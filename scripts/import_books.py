import os
import re
import sys
import argparse
import unicodedata
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('csv_path')
ap.add_argument('--source', default='import', help="short name of where the list comes from, e.g. openstax")
ap.add_argument('--data_dir', default='data/processed')
ap.add_argument('--no_check', action='store_true', help="do not test the links")
args = ap.parse_args()

HEAD = {'User-Agent': 'SmartLibrary-student-project'}


def fold(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return s.lower()


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', fold(s))).strip()


def link_ok(url):
    if not url:
        return True
    try:
        r = requests.get(url, headers=HEAD, timeout=12, stream=True)
        r.close()
        return r.status_code < 400
    except Exception:
        return False


df = pd.read_csv(args.csv_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
for col in ['title', 'author', 'read_url']:
    if col not in df.columns:
        sys.exit(f"The CSV needs a '{col}' column. Found: {list(df.columns)}")
for col in ['categories', 'description', 'cover_url']:
    if col not in df.columns:
        df[col] = ''
df = df.apply(lambda c: c.str.strip())
df = df[df['title'] != ''].copy()
df['author'] = df['author'].replace('', 'Unknown author')
print(f"Rows in the CSV: {len(df)}")

if not args.no_check:
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(link_ok, df['read_url']))
    bad = df[[not o for o in ok]]
    if len(bad):
        print(f"\nSkipped {len(bad)} books whose link did not open (fix them in the CSV and run again):")
        for _, r in bad.iterrows():
            print("  -", r['title'], "->", r['read_url'])
    df = df[ok].copy()

slug = lambda r: re.sub(r'[^a-z0-9]+', '-', fold(r['read_url'].rstrip('/').split('/')[-1] or r['title'])).strip('-')
df['isbn13'] = [f"{args.source}:{slug(r)}" for _, r in df.iterrows()]      # an id that is not a real ISBN
df['source'] = 'import_' + args.source
df['title_norm'] = df['title'].map(norm)
df['author_norm'] = df['author'].map(norm)
df['mood'] = ''
df['mood_confidence'] = 0.0
df['rating_count'] = 0
df['embedding_text'] = df['title'] + ' by ' + df['author'] + '. ' + df['categories'] + '. ' + df['description']
df = df.drop_duplicates('isbn13')

out_rows = f'{args.data_dir}/extra_books.parquet'
out_vecs = f'{args.data_dir}/extra_embeddings.npy'
old, old_vecs = None, None
if os.path.exists(out_rows) and os.path.exists(out_vecs):
    old = pd.read_parquet(out_rows)
    old_vecs = np.load(out_vecs)
    if len(old) != len(old_vecs):                 # out of sync: rebuild everything from scratch
        print("extra_books and extra_embeddings were out of sync: re-embedding all extra books.")
        old_vecs = None
updated = int(old['isbn13'].isin(df['isbn13']).sum()) if old is not None else 0

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Computing embeddings for {len(df)} books (this can take a few minutes for big lists)...")
new_vecs = np.asarray(model.encode(df['embedding_text'].tolist(), batch_size=128, show_progress_bar=len(df) > 2000), dtype='float32')

if old is not None:
    keep = ~old['isbn13'].isin(df['isbn13']).values      # old rows not replaced by this import
    kept_rows = old[keep]
    kept_vecs = old_vecs[keep] if old_vecs is not None else np.asarray(
        model.encode(kept_rows['embedding_text'].tolist(), batch_size=128), dtype='float32')
    allrows = pd.concat([kept_rows, df], ignore_index=True)
    vecs = np.vstack([kept_vecs, new_vecs])
else:
    allrows, vecs = df.reset_index(drop=True), new_vecs

allrows.to_parquet(out_rows, index=False)
np.save(out_vecs, vecs.astype('float32'))
print(f"\nDONE. Added {len(df) - updated}, updated {updated}. Extra books in total: {len(allrows)}")
print("Restart the app (py app.py) to see them.")