"""Build CSV lists of free, legal books from open sources, ready for scripts/import_books.py.

Run from D:/documents/Smart_Library_V2:
    python scripts/fetch_sources.py gutenberg        # all English Project Gutenberg books (uses data/raw/pg_catalog.csv)
    python scripts/fetch_sources.py opentextbooks    # Open Textbook Library (open.umn.edu): free university textbooks
Then import each list:
    python scripts/import_books.py data/sources/gutenberg.csv --source gutenberg --no_check
    python scripts/import_books.py data/sources/opentextbooks.csv --source otl --no_check

Every source here is public domain or openly licensed, so the "Read" links are legal to share.
"""
import ast
import json
import os
import re
import sys
import time

import pandas as pd
import requests

OUT_DIR = os.path.join('data', 'sources')
HEAD = {'User-Agent': 'SmartLibraryStudentProject/1.0'}


def clean_author(a):
    """'Austen, Jane, 1775-1817; Someone' -> 'Jane Austen' (first author only)."""
    first = str(a).split(';')[0].strip()
    first = first.split('[')[0].strip()                        # drop "[Editor]" style roles
    parts = [p.strip() for p in first.split(',')]
    parts = [p for p in parts if p and not any(ch.isdigit() for ch in p)]
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}"
    return parts[0] if parts else 'Unknown author'


def gutenberg():
    cat = pd.read_csv(os.path.join('data', 'raw', 'pg_catalog.csv'))
    is_en = cat['Language'].fillna('').str.split(';').apply(lambda x: 'en' in [s.strip() for s in x])
    cat = cat[(cat['Type'] == 'Text') & is_en & cat['Title'].notna()].copy()

    shelves = cat['Bookshelves'].fillna('').str.replace('Category: ', '', regex=False)
    subjects = cat['Subjects'].fillna('').str.replace(' -- ', ', ', regex=False)
    out = pd.DataFrame({
        'title': cat['Title'].astype(str).str.split(r'[\r\n]+').str[0].str.strip(),
        'author': cat['Authors'].fillna('').map(clean_author),
        'read_url': 'https://www.gutenberg.org/ebooks/' + cat['Text#'].astype(str),
        'categories': shelves.str.slice(0, 300),
        'description': ('Subjects: ' + subjects).str.slice(0, 600),
        'cover_url': 'https://www.gutenberg.org/cache/epub/' + cat['Text#'].astype(str) + '/pg' + cat['Text#'].astype(str) + '.cover.medium.jpg',
    })
    out = out[out['title'] != '']
    return out, 'gutenberg.csv'


def _names(raw):
    try:
        return ast.literal_eval(raw) if isinstance(raw, str) else (raw or [])
    except (ValueError, SyntaxError):
        return []


def _otl_page(page, cache_dir):
    """One API page, cached on disk so an interrupted run resumes where it stopped."""
    path = os.path.join(cache_dir, f'page_{page}.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    url = f'https://open.umn.edu/opentextbooks/textbooks.json?page={page}'
    for attempt in range(6):
        try:
            r = requests.get(url, headers=HEAD, timeout=30)
            if r.status_code == 200:
                data = r.json()
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                time.sleep(1.0)                           # be polite to the server
                return data
            print(f"  page {page}: HTTP {r.status_code}, waiting...")
        except (requests.exceptions.RequestException, ValueError):
            print(f"  page {page}: connection problem, waiting...")
        time.sleep(min(15 * 2 ** attempt, 180))           # 15s, 30s, 60s, 120s, 180s
    sys.exit(f"Stopped at page {page}. Pages so far are saved; run the same command again later to continue.")


def opentextbooks():
    cache_dir = os.path.join(OUT_DIR, 'otl_pages')
    os.makedirs(cache_dir, exist_ok=True)
    rows, page, url = [], 0, True
    while url:
        page += 1
        data = _otl_page(page, cache_dir)
        for b in data.get('data', []):
            if b.get('language') not in (None, 'eng'):
                continue
            people = _names(b.get('contributors'))
            authors = [' '.join(x for x in (p.get('first_name'), p.get('middle_name'), p.get('last_name')) if x)
                       for p in people if isinstance(p, dict) and p.get('contributor_type', 'Author') in ('Author', None)]
            subjects = [s.get('name', '') for s in _names(b.get('subjects')) if isinstance(s, dict)]
            rows.append({
                'title': (b.get('title') or '').strip(),
                'author': ', '.join(a for a in authors[:3] if a) or 'Unknown author',
                'read_url': b.get('url') or '',
                'categories': 'Textbook; ' + '; '.join(subjects),
                'description': (b.get('description') or '').strip()[:1500],
                'cover_url': '',
            })
        url = data.get('links', {}).get('next')
        if page % 20 == 0:
            print(f"  page {page}: {len(rows)} textbooks so far")
    out = pd.DataFrame(rows)
    return out[(out['title'] != '') & (out['read_url'] != '')], 'opentextbooks.csv'


def textbook_pdfs():
    """Textbooks whose PDF link points straight at a .pdf file (those can be shown inside our reader).
    Uses the pages already saved by `opentextbooks`; nothing is downloaded."""
    cache_dir = os.path.join(OUT_DIR, 'otl_pages')
    rows = []
    for name in sorted(os.listdir(cache_dir)):
        with open(os.path.join(cache_dir, name), encoding='utf-8') as f:
            for b in json.load(f).get('data', []):
                for x in (b.get('formats') or []):
                    url = x.get('url') if isinstance(x, dict) and x.get('type') == 'PDF' else ''
                    if url and re.search(r'\.pdf($|[?#])', url, re.I) and b.get('url'):
                        rows.append({'read_url': b['url'], 'title': (b.get('title') or '').strip(), 'pdf_url': url})
                        break
    return pd.DataFrame(rows), 'textbook_pdfs.csv'


if __name__ == '__main__':
    sources = {'gutenberg': gutenberg, 'opentextbooks': opentextbooks, 'textbook_pdfs': textbook_pdfs}
    if len(sys.argv) != 2 or sys.argv[1] not in sources:
        sys.exit('Usage: python scripts/fetch_sources.py ' + '|'.join(sources))
    df, name = sources[sys.argv[1]]()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.drop_duplicates('read_url').to_csv(path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} books -> {path}")
