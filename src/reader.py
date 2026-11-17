"""In-site reader for Project Gutenberg books (public domain).

The plain text is downloaded once per book and cached in data/cache/gutenberg/<id>.txt,
then cleaned (Gutenberg header/footer removed) and split into pages of paragraphs.
"""
import os
import re
import threading
import requests

CACHE_DIR = os.path.join('data', 'cache', 'gutenberg')
GUTENBERG_ID = re.compile(r'gutenberg\.org/ebooks/(\d+)')
PARAS_PER_PAGE = 40
MAX_BYTES = 8 * 1024 * 1024          # skip absurdly large files


class ReaderError(Exception):
    """The book text could not be downloaded right now."""


def gutenberg_id(read_url):
    """'https://www.gutenberg.org/ebooks/11' -> 11, anything else -> None."""
    m = GUTENBERG_ID.search(read_url or '')
    return int(m.group(1)) if m else None


# Official Project Gutenberg mirrors, tried in order when www.gutenberg.org is down (it often is for maintenance).
MIRRORS = ['http://mirrors.xmission.com/gutenberg',
           'https://mirror.csclub.uwaterloo.ca/gutenberg',
           'https://gutenberg.nabasny.com']


def mirror_dir(gid):
    """Mirror folder layout: one folder per digit except the last, then the id.
    1342 -> 1/3/4/1342   11 -> 1/11   5 -> 0/5"""
    s = str(int(gid))
    return '/'.join(list(s[:-1]) + [s]) if len(s) > 1 else f'0/{s}'


def candidate_urls(gid):
    """(url, encoding, timeout) in the order to try."""
    gid = int(gid)
    out = [(f'https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt', 'utf-8', 5),
           (f'https://www.gutenberg.org/ebooks/{gid}.txt.utf-8', 'utf-8', 5)]
    for base in MIRRORS:
        d = f'{base}/{mirror_dir(gid)}'
        out += [(f'{d}/{gid}-0.txt', 'utf-8', 12), (f'{d}/{gid}.txt', 'auto', 12), (f'{d}/{gid}-8.txt', 'latin-1', 12)]
    out.append((f'https://gutenberg.pglaf.org/cache/epub/{gid}/pg{gid}.txt', 'utf-8', 40))   # works but slow
    return out


def _decode(raw, encoding):
    if encoding == 'auto':
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')
    return raw.decode(encoding, errors='replace')


def _download(gid, user_agent):
    for url, encoding, timeout in candidate_urls(gid):
        try:
            r = requests.get(url, headers={'User-Agent': user_agent}, timeout=timeout)
        except requests.exceptions.RequestException:
            continue
        if r.status_code != 200 or not (0 < len(r.content) <= MAX_BYTES):
            continue
        text = _decode(r.content, encoding)
        if text.lstrip()[:1] == '<':          # an HTML error page, not the book (mirrors don't always send text/plain)
            continue
        return text
    raise ReaderError()


def get_text(gid, user_agent):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{int(gid)}.txt')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return f.read()
    text = _download(gid, user_agent)
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'    # write aside, then swap in: never a half file
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)
    return text


def clean(text):
    """Drop the Project Gutenberg licence header and footer."""
    text = text.replace('\r\n', '\n')
    start = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n', text, re.I)
    if start:
        text = text[start.end():]
    end = re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text, re.I)
    if end:
        text = text[:end.start()]
    return text.strip()


CHAPTER = re.compile(r'^(chapter|book|part|act|canto|letter)\b[\s\w.:-]{0,60}$|^[IVXLC]+\.?$', re.I)


def paragraphs(text):
    """Split into paragraphs; lines inside a paragraph are joined. Headings are marked."""
    out = []
    for block in re.split(r'\n\s*\n', text):
        line = ' '.join(block.split())
        if not line:
            continue
        out.append({'text': line, 'heading': bool(CHAPTER.match(line)) and len(line) < 70})
    return out


def load_book(gid, user_agent):
    """-> list of pages, each a list of paragraph dicts."""
    paras = paragraphs(clean(get_text(gid, user_agent)))
    pages, cur = [], []
    for p in paras:
        if p['heading'] and len(cur) >= 8:        # start a new page at a chapter heading
            pages.append(cur)
            cur = []
        cur.append(p)
        if len(cur) >= PARAS_PER_PAGE:
            pages.append(cur)
            cur = []
    if cur:
        pages.append(cur)
    return pages or [[{'text': 'This book has no readable text.', 'heading': False}]]
