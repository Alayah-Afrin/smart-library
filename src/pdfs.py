"""Free textbook PDFs shown inside our own reader page.

Only PDFs from data/sources/textbook_pdfs.csv (made by `python scripts/fetch_sources.py textbook_pdfs`)
can be opened, so this can't be used to fetch anything else. Each PDF is downloaded once, checked to really
be a PDF, and kept in data/cache/pdf/ (named after its address).
"""
import hashlib
import os
import threading

import pandas as pd
import requests

LIST_PATH = os.path.join('data', 'sources', 'textbook_pdfs.csv')
CACHE_DIR = os.path.join('data', 'cache', 'pdf')
MAX_BYTES = 150 * 1024 * 1024          # skip anything bigger than 150 MB
CHUNK = 256 * 1024


class PdfError(Exception):
    """The PDF could not be fetched right now (site down, not a PDF, too big)."""


def load_list():
    """-> (list of dicts with title/pdf_url, {read_url: pdf id})"""
    if not os.path.exists(LIST_PATH):
        return [], {}
    df = pd.read_csv(LIST_PATH, dtype=str).fillna('')
    df = df[(df['pdf_url'] != '') & (df['read_url'] != '')]
    if 'works' in df:                      # made by scripts/check_pdfs.py: keep only links that really open
        df = df[df['works'] == 'yes']
    df = df.drop_duplicates('read_url').reset_index(drop=True)
    books = df[['title', 'pdf_url', 'read_url']].to_dict('records')
    return books, {b['read_url']: i for i, b in enumerate(books)}


def cached_path(url):
    """File name from the PDF's address, so a saved PDF always belongs to the same book."""
    return os.path.join(CACHE_DIR, hashlib.sha1(url.encode('utf-8')).hexdigest()[:20] + '.pdf')


def ensure_pdf(url, user_agent):
    """Download the PDF once (checked + size-capped), return the local file path."""
    path = cached_path(url)
    if os.path.exists(path):
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        with requests.get(url, headers={'User-Agent': user_agent}, timeout=30, stream=True) as r:
            if r.status_code != 200:
                raise PdfError(f'HTTP {r.status_code}')
            size, first = 0, True
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(CHUNK):
                    if first:
                        if not chunk.lstrip()[:5].startswith(b'%PDF'):
                            raise PdfError('not a PDF')       # e.g. a web page or a login screen
                        first = False
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise PdfError('too big')
                    f.write(chunk)
            if first:
                raise PdfError('empty')
        os.replace(tmp, path)                                 # never leave a half file under the real name
        return path
    except requests.exceptions.RequestException as e:
        raise PdfError(type(e).__name__) from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
