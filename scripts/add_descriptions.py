
import sys
import os
import re
import time
import argparse
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

sys.path.append('src')
from search import make_dedup_key

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data/processed')
ap.add_argument('--limit', type=int, default=3000)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()

OL = 'https://openlibrary.org'
HEAD = {'User-Agent': 'SmartLibrary-student-project'}


class Fail(Exception):
    """Open Library did not answer properly (blocked, busy or offline). NOT the same as 'no description'."""


def get_json(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=10)
        except Exception:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(3 * (attempt + 1))          # 429 / 5xx: wait, then try again
    raise Fail()


def clean(text):
    text = re.sub(r'\[\d+\]|\(\[source\][^)]*\)', '', text)
    text = re.sub(r'\s*-{3,}.*$', '', text, flags=re.S)       # drop "-----" footers with links
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 40 else ''


def fetch_description(title, author, isbn13):
    work_key = None
    if isbn13:
        ed = get_json(f'{OL}/isbn/{isbn13}.json')
        if ed and ed.get('works'):
            work_key = ed['works'][0]['key']
            d = ed.get('description')
            d = d.get('value') if isinstance(d, dict) else d
            if d and clean(d):
                return clean(d)
    if not work_key:
        s = get_json(f'{OL}/search.json', {'title': title, 'author': author, 'limit': 1, 'fields': 'key'})
        if s and s.get('docs'):
            work_key = s['docs'][0]['key']
    if not work_key:
        return ''
    w = get_json(f'{OL}{work_key}.json')
    d = (w or {}).get('description')
    d = d.get('value') if isinstance(d, dict) else d
    return clean(d) if d else ''


if __name__ == '__main__':
    out_path = f'{args.data_dir}/descriptions.parquet'
    books = pd.read_parquet(f'{args.data_dir}/books_search_ready.parquet')
    books['description'] = books['description'].fillna('')
    books['author'] = books['author'].fillna('')
    books['dedup_key'] = [make_dedup_key(t, a) for t, a in zip(books['title'], books['author'])]
    isbn_col = 'isbn13' if 'isbn13' in books.columns else None

    done = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['dedup_key', 'description'])
    todo = books[(books['description'].str.len() < 40) & ~books['dedup_key'].isin(set(done['dedup_key']))]
    todo = todo.drop_duplicates('dedup_key')

    gl = f'{args.data_dir}/gutenberg_links.parquet'
    if os.path.exists(gl):                                   # free books first
        free = set(pd.read_parquet(gl)['dedup_key'])
        todo = pd.concat([todo[todo['dedup_key'].isin(free)], todo[~todo['dedup_key'].isin(free)]])
    todo = todo.head(args.limit)
    print(f"Already done: {len(done)} | fetching now: {len(todo)}")

    rows = [(r.dedup_key, r.title, r.author, str(getattr(r, isbn_col)) if isbn_col and pd.notna(getattr(r, isbn_col)) else '')
            for r in todo.itertuples()]

    def safe(x):
        try:
            return fetch_description(x[1], x[2], x[3])
        except Fail:
            return None                         # failed request: do NOT save, retry next run

    new, found, failed = [], 0, 0
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            for i, ((key, *_), desc) in enumerate(zip(rows, ex.map(safe, rows)), 1):
                if desc is None:
                    failed += 1
                else:
                    new.append({'dedup_key': key, 'description': desc})
                    found += bool(desc)
                if i % 100 == 0:
                    print(f"  {i}/{len(rows)}  found so far: {found}  failed requests: {failed}")
                    pd.concat([done, pd.DataFrame(new)]).to_parquet(out_path, index=False)
                if failed >= 50 and failed > i * 0.5:
                    print("Open Library is blocking or busy. Stop here and try again later.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    except KeyboardInterrupt:
        print("Stopped by you. Saving what we have...")
    pd.concat([done, pd.DataFrame(new)]).to_parquet(out_path, index=False)
    print(f"DONE. Saved {len(new)} books, found a description for {found}, {failed} failed requests (retried next run). File: {out_path}")