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


def openstax():
    """Every live English OpenStax textbook (free, peer-reviewed, CC BY)."""
    url = 'https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&fields=*&limit=250'
    items = requests.get(url, headers=HEAD, timeout=60).json()['items']
    rows = []
    for b in items:
        meta = b.get('meta') or {}
        if b.get('book_state') != 'live' or meta.get('locale', 'en') != 'en':
            continue
        subjects = [s.get('subject_name', '') for s in b.get('book_subjects') or [] if isinstance(s, dict)]
        rows.append({
            'title': (b.get('title') or '').strip(),
            'author': 'OpenStax',
            'read_url': meta.get('html_url') or f"https://openstax.org/details/books/{meta.get('slug', '')}",
            'categories': 'Textbook; ' + '; '.join(s for s in subjects if s),
            'description': (meta.get('search_description') or '').strip()[:1500],
            'cover_url': b.get('cover_url') or '',
        })
    out = pd.DataFrame(rows)
    return out[out['title'] != ''], 'openstax_full.csv'


FPB = 'https://raw.githubusercontent.com/EbookFoundation/free-programming-books/main/books/'
ENTRY = re.compile(r'^\s*[*-]\s+\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)(?P<rest>.*)$')


def programming():
    """The free-programming-books lists (EbookFoundation, CC BY 4.0): free CS and programming books,
    by subject and by programming language. The meta-lists of other sites are skipped."""
    rows = []
    for name in ('free-programming-books-subjects.md', 'free-programming-books-langs.md'):
        text = requests.get(FPB + name, headers=HEAD, timeout=60).text
        heading = sub = ''
        for line in text.splitlines():
            if line.startswith('### '):
                heading, sub = line[4:].strip(), ''
                continue
            if line.startswith('#### '):
                sub = line[5:].strip()
                continue
            m = ENTRY.match(line)
            if not m or not heading or heading.lower().startswith(('0 - meta', 'index')):
                continue
            rest = m.group('rest')
            author = ''
            am = re.match(r'\s*-\s*([^(]+?)\s*(\(|$)', rest)
            if am:
                author = re.sub(r'\*', '', am.group(1)).strip(' ,.')
            fmt = re.findall(r'\((HTML|PDF|EPUB|Markdown|LaTeX)\)', rest)
            topic = '; '.join(x for x in (heading, sub) if x)
            rows.append({
                'title': m.group('title').strip(),
                'author': author if author and len(author) < 120 else 'Unknown author',
                'read_url': m.group('url'),
                'categories': 'Computer Science; Programming; ' + topic,
                'description': f"Free book about {topic}." + (f" Formats: {', '.join(fmt)}." if fmt else ''),
                'cover_url': '',
            })
    out = pd.DataFrame(rows)
    return out[out['title'].str.len() >= 3], 'programming.csv'


def doab():
    """Directory of Open Access Books: peer-reviewed open-access academic books (mostly CC licences).
    Uses data/raw/doab_export.csv (the site's full CSV export). English books only."""
    df = pd.read_csv(os.path.join('data', 'raw', 'doab_export.csv'), dtype=str, keep_default_na=False,
                     on_bad_lines='skip', low_memory=False)
    col = lambda *names: next((c for c in names if c in df.columns), None)
    c_title, c_auth, c_editor = col('dc.title'), col('dc.contributor.author'), col('dc.contributor.editor')
    c_lang, c_subj, c_kw = col('dc.language'), col('dc.subject.classification'), col('dc.subject.other')
    c_abs = col('dc.description.abstract')
    c_dl, c_uri = col('oapen.identifier.downloadUrl', 'BITSTREAM Download URL'), col('dc.identifier.uri')
    if not (c_title and c_lang and (c_dl or c_uri)):
        sys.exit(f'Unexpected DOAB columns: {list(df.columns)[:60]}')
    eng = df[df[c_lang].str.contains(r'(?i)\b(?:english|eng|en)\b', regex=True)]
    empty = pd.Series('', index=eng.index)
    get = lambda c: eng[c] if c else empty

    def first(v):
        return str(v).split('||')[0].strip()

    def subject_names(v):
        """'thema EDItEUR::U Computing and IT::UY Computer science' -> 'Computing and IT; Computer science'."""
        names = []
        for s in str(v).split('||'):
            for part in s.split('::')[1:] or [s]:
                name = re.sub(r'^[A-Z0-9]{1,6}\s+', '', part.strip())     # drop the subject code ("UY ")
                if name and name not in names and not name.lower().startswith('thema'):
                    names.append(name)
        return '; '.join(names)[:300]

    def person(v):
        p = first(v)
        if ',' in p:                                     # "Smith, John" -> "John Smith"
            last, _, given = p.partition(',')
            p = f'{given.strip()} {last.strip()}'.strip()
        return p

    out = pd.DataFrame({
        'title': get(c_title).map(first),
        'author': [person(a) or person(e) or 'Unknown author' for a, e in zip(get(c_auth), get(c_editor))],
        'read_url': [first(u) if first(u).startswith('http') else first(h) for u, h in zip(get(c_uri), get(c_dl))],
        'categories': 'Academic; ' + get(c_subj).map(subject_names),
        'description': (get(c_abs).str.slice(0, 1200) + ' Keywords: '
                        + get(c_kw).str.replace('||', ', ', regex=False).str.slice(0, 200)),
        'cover_url': '',
    })
    return out[(out['title'] != '') & out['read_url'].str.startswith('http')], 'doab.csv'


MODERN_SUBJECTS = [
    'fiction', 'fantasy', 'science fiction', 'romance', 'mystery', 'thriller', 'horror', 'young adult fiction',
    'juvenile fiction', 'children', 'graphic novels', 'historical fiction', 'humor', 'poetry', 'biography',
    'history', 'science', 'computer science', 'programming', 'mathematics', 'engineering', 'physics',
    'biology', 'chemistry', 'medicine', 'psychology', 'philosophy', 'economics', 'business',
    'self-help', 'politics', 'religion', 'art', 'music', 'cooking', 'travel', 'sports', 'education',
]


def _ol_page(subject, page, cache_dir):
    """One Open Library search page (1,000 books), cached on disk so a stopped run resumes."""
    path = os.path.join(cache_dir, f"{re.sub(r'[^a-z0-9]+', '_', subject)}_{page}.json")
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    params = {'q': f'subject:"{subject}" first_publish_year:[2005 TO 2026] language:eng', 'sort': 'readinglog',
              'fields': 'key,title,author_name,first_publish_year,ratings_average,ratings_count,'
                        'readinglog_count,cover_i,subject', 'limit': 1000, 'page': page}
    for attempt in range(5):
        try:
            r = requests.get('https://openlibrary.org/search.json', params=params, headers=HEAD, timeout=120)
            if r.status_code == 200:
                data = r.json()
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                time.sleep(2)                              # be polite to Open Library
                return data
            print(f"  {subject} p{page}: HTTP {r.status_code}, waiting...")
        except (requests.exceptions.RequestException, ValueError):
            print(f"  {subject} p{page}: connection problem, waiting...")
        time.sleep(20 * (attempt + 1))
    print(f"  {subject} p{page}: skipped after 5 tries")
    return {'docs': []}


def modern():
    """Popular books first published 2005-2026 (Open Library), so newer books such as Percy Jackson and
    The Hunger Games can be found. These are NOT free to read: read_url stays empty and the page offers
    'Read or borrow' on Open Library. Open Library's own reader ratings are kept (ol_avg / ol_count)."""
    cache_dir = os.path.join(OUT_DIR, 'ol_modern_pages')
    os.makedirs(cache_dir, exist_ok=True)
    rows, seen = {}, set()
    for subject in MODERN_SUBJECTS:
        for page in range(1, 4):                           # up to 3,000 most-read books per subject
            docs = _ol_page(subject, page, cache_dir).get('docs', [])
            for d in docs:
                key = d.get('key')
                if not key or key in seen or (d.get('readinglog_count') or 0) < 15:
                    continue
                seen.add(key)
                subs = [s for s in (d.get('subject') or [])[:12] if len(s) < 40]
                rows[key] = {
                    'title': (d.get('title') or '').strip(),
                    'author': (d.get('author_name') or ['Unknown author'])[0],
                    'read_url': '',
                    'ol_key': key,
                    'categories': '; '.join(dict.fromkeys([subject.title()] + subs))[:300],
                    'description': f"First published {d.get('first_publish_year', '')}. Subjects: {', '.join(subs[:8])}.",
                    'cover_url': f"https://covers.openlibrary.org/b/id/{d['cover_i']}-M.jpg" if d.get('cover_i') else '',
                    'ol_avg': d.get('ratings_average') or '',
                    'ol_count': d.get('ratings_count') or 0,
                    'rating_count': d.get('readinglog_count') or 0,
                }
            if len(docs) < 1000 or (docs and (docs[-1].get('readinglog_count') or 0) < 15):
                break
        print(f"  {subject}: {len(rows)} books so far", flush=True)
    out = pd.DataFrame(list(rows.values()))
    return out[out['title'] != ''], 'modern.csv'


if __name__ == '__main__':
    sources = {'gutenberg': gutenberg, 'opentextbooks': opentextbooks, 'textbook_pdfs': textbook_pdfs,
               'openstax': openstax, 'programming': programming, 'doab': doab, 'modern': modern}
    if len(sys.argv) != 2 or sys.argv[1] not in sources:
        sys.exit('Usage: python scripts/fetch_sources.py ' + '|'.join(sources))
    df, name = sources[sys.argv[1]]()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    linked = df['read_url'].fillna('') != ''               # books with no free link (modern) are all kept
    df = pd.concat([df[linked].drop_duplicates('read_url'), df[~linked]], ignore_index=True)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} books -> {path}")
