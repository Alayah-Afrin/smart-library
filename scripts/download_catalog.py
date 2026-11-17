"""Download Project Gutenberg's catalog, resuming if the connection drops.
Run from D:/documents/Smart_Library:  python scripts/download_catalog.py
"""
import os
import time
import requests

URL = 'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv'
OUT = os.path.join('data', 'raw', 'pg_catalog.csv')
HEADERS = {'User-Agent': 'SmartLibraryStudentProject/1.0'}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
try:
    total = int(requests.head(URL, headers=HEADERS, timeout=30, allow_redirects=True).headers.get('Content-Length', 0))
except requests.exceptions.RequestException:
    total = 0                                          # size unknown: we'll just try the download
print('Expected size:', total // 1024, 'KB')

for attempt in range(1, 31):
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if total and have >= total:
        break
    try:
        headers = dict(HEADERS, Range=f'bytes={have}-') if have else HEADERS
        with requests.get(URL, headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 200 and have:          # server ignored Range: start over
                have = 0
            mode = 'ab' if have else 'wb'
            with open(OUT, mode) as f:
                for chunk in r.iter_content(64 * 1024):
                    f.write(chunk)
        if not total:                                  # size unknown and no error: finished
            break
    except requests.exceptions.RequestException as e:
        print(f'attempt {attempt}: connection dropped ({type(e).__name__}), retrying...')
        time.sleep(2)
    print('have', os.path.getsize(OUT) // 1024, 'KB')

size = os.path.getsize(OUT)
print('DONE' if (not total or size >= total) else 'INCOMPLETE', size // 1024, 'KB ->', OUT)
