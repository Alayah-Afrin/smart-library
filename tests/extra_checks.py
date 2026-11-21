"""Extra checks: heavy typos (3-4 wrong letters in a word), typos inside whole sentences, shortcuts, names.
Run from D:/documents/Smart_Library_V2:   python tests/extra_checks.py
A search passes when one of the TOP 10 books matches the expected pattern (title, author or category)."""
import re
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import search as S
S.live_lookup = lambda *a, **k: []
S.get_better_cover = lambda isbn, title, author, fallback: fallback

CASES = [
    # heavy typos: 3-4 letters wrong in one word
    ('Heavy typo', 'hary pottr', r'harry potter'),
    ('Heavy typo', 'harrry pottter', r'harry potter'),
    ('Heavy typo', 'stefen kingg', r'stephen king'),
    ('Heavy typo', 'stfn kng', r'stephen king'),
    ('Heavy typo', 'agtha chrstie', r'christie'),
    ('Heavy typo', 'shakspere', r'shakespeare'),
    ('Heavy typo', 'tolkein lrd of the rngs', r'lord of the rings|tolkien'),
    ('Heavy typo', 'pythn progrmming', r'python'),
    ('Heavy typo', 'mchine lernng', r'machine learning'),
    ('Heavy typo', 'artifcal inteligence', r'artificial intelligence'),
    ('Heavy typo', 'algorithmz and datastructurs', r'algorithm|data structure'),
    ('Heavy typo', 'dostoevky', r'dostoevsk|dostoyevsk'),
    ('Heavy typo', 'hemingwy', r'hemingway'),
    ('Heavy typo', 'frankenstn', r'frankenstein'),
    # typos inside whole sentences
    ('Sentence typo', 'a book abuot a young wizzard at a magic scool', r'wizard|harry potter|magic'),
    ('Sentence typo', 'i want to lern pyhton programing for beginers', r'python'),
    ('Sentence typo', 'best mistery novells by agatha cristie', r'christie|mystery'),
    ('Sentence typo', 'books on the histroy of anceint rome', r'rome|roman'),
    ('Sentence typo', 'somthing about artifical inteligence and robots', r'artificial|robot'),
    ('Sentence typo', 'storys for chilldren about animls', r'child|juvenile|animal'),
    ('Sentence typo', 'a romantic novle by jane austin', r'austen'),
    # shortcuts
    ('Shortcut', 'ai', r'artificial intelligence'),
    ('Shortcut', 'ml', r'machine learning'),
    ('Shortcut', 'dl', r'deep learning'),
    ('Shortcut', 'nlp', r'natural language'),
    ('Shortcut', 'dsa', r'data structures'),
    ('Shortcut', 'os', r'operating system'),
    ('Shortcut', 'db', r'database'),
    ('Shortcut', 'dbms', r'database'),
    ('Shortcut', 'cv', r'computer vision|vision'),
    ('Shortcut', 'oop', r'object.oriented'),
    ('Shortcut', 'sql', r'sql'),
    ('Shortcut', 'lotr', r'lord of the rings'),
    ('Shortcut', 'hp', r'harry potter'),
    ('Shortcut', 'got', r'game of thrones'),
    # finding by name: full, short, initials, nicknames
    ('Name', 'stephen king', r'stephen king'),
    ('Name', 'king', r'stephen king'),
    ('Name', 'jk rowling', r'rowling'),
    ('Name', 'rowling', r'rowling'),
    ('Name', 'tolkien', r'tolkien'),
    ('Name', 'christie', r'christie'),
    ('Name', 'mark twain', r'twain'),
    ('Name', 'sam clemens', r'twain|clemens'),
    ('Name', 'dr seuss', r'seuss'),
    ('Name', 'jane austen', r'austen'),
    ('Name', 'george orwell', r'orwell'),
    ('Name', 'knuth', r'knuth'),
    ('Name', 'cormen', r'cormen|algorithms'),
]


def blob(df, n=10):
    return [S.fold(f"{r.get('title', '')} | {r.get('author', '')} | {r.get('categories', '')}")
            for _, r in df.head(n).fillna('').iterrows()]


if __name__ == '__main__':
    system = S.load_search_system('data/processed')
    bad = []
    stats = {}
    for kind, q, pat in CASES:
        df, banner = S.search_books(q, system, top_k=24)
        ok = len(df) > 0 and any(re.search(pat, t) for t in blob(df))
        if not ok and kind == 'Shortcut':                 # an ambiguous short form ("hp"): the famous meaning is a link
            links = (df.attrs.get('suggestions') or []) + (df.attrs.get('alternatives') or [])
            ok = any(re.search(pat, S.fold(s)) for s in links)
        s = stats.setdefault(kind, [0, 0])
        s[0] += ok
        s[1] += 1
        if not ok:
            top = blob(df, 3)
            bad.append(f"  [{kind}] {q!r}\n      message: {banner or '-'}\n      top: {' || '.join(t.split(' | ')[0][:40] for t in top) or '(nothing)'}")
    # learning path: several subjects, time limits, things the student already knows
    import paths
    PATHS = [
        ('Learning path', 'i want to learn pythoin and ml at the same time, within 1 month',
         lambda p: p['fast_track'] and p['weeks'] == 4 and 'programming' in p['subject'] and 'machine learning' in p['subject']),
        ('Learning path', 'learn python and ml within 2 months', lambda p: p['weeks'] == 8 and p['fast_track']),
        ('Learning path', 'learn machine learning in 6 weeks', lambda p: p['weeks'] == 6),
        ('Learning path', 'machine learning from scratch', lambda p: p['weeks'] == 23 and not p['fast_track']),
        ('Learning path', 'I already know python and want to learn web development',
         lambda p: p['subject'] == 'web development' and p['steps'][0]['status'] == 'known'),
        ('Learning path', 'neural networks', lambda p: p['subject'] == 'deep learning'),
        ('Learning path', 'economics', lambda p: len(p['steps']) == 3),
        ('Learning path', 'asdfghjk', lambda p: 'error' in p),
    ]
    for kind, goal, check in PATHS:
        try:
            ok = bool(check(paths.plan(goal, system)))
        except Exception:
            ok = False
        s = stats.setdefault(kind, [0, 0])
        s[0] += ok
        s[1] += 1
        if not ok:
            bad.append(f"  [{kind}] {goal!r}")
    total = sum(v[0] for v in stats.values())
    n_cases = len(CASES) + len(PATHS)
    print(f"\n{total}/{n_cases} extra checks passed")
    for k, (a, b) in stats.items():
        print(f"  {'OK  ' if a == b else 'FAIL'} {k:16} {a}/{b}")
    if bad:
        print('\n=== need a look ===')
        print('\n'.join(bad))
