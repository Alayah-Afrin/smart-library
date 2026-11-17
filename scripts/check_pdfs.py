"""Check which textbook PDF links really open a PDF (reads only the first 2 KB of each file).
Run from D:/documents/Smart_Library_V2:   python scripts/check_pdfs.py
Updates data/sources/textbook_pdfs.csv with a 'works' column; only working ones get "Read here" on the site.
Publishers move files, so re-run this now and then."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

PATH = os.path.join('data', 'sources', 'textbook_pdfs.csv')
HEAD = {'User-Agent': 'SmartLibraryStudentProject/1.0'}


def works(url):
    try:
        with requests.get(url, headers=HEAD, timeout=25, stream=True) as r:
            if r.status_code != 200:
                return False
            first = next(r.iter_content(2048), b'')
            return first.lstrip()[:5].startswith(b'%PDF')
    except Exception:
        return False


if __name__ == '__main__':
    if not os.path.exists(PATH):
        sys.exit('Run  python scripts/fetch_sources.py textbook_pdfs  first.')
    df = pd.read_csv(PATH, dtype=str).fillna('')
    with ThreadPoolExecutor(8) as ex:
        df['works'] = ['yes' if ok else 'no' for ok in ex.map(works, df['pdf_url'])]
    df.to_csv(PATH, index=False, encoding='utf-8')
    print(f"{(df['works'] == 'yes').sum()} of {len(df)} PDF links work -> {PATH}")
