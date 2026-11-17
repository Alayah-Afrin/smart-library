"""Run from D:/documents/Smart_Library with:  python tests/check_search.py
Prints PASS/FAIL for each check. Takes about as long as app startup, then a few seconds."""
import sys
sys.path.append('src')

from search import load_search_system, search_books, fold

system = load_search_system('data/processed')
wd = system['work_dedup']

results_log = []


def run(q, cat=None):
    df, banner = search_books(q, system, category_filter=cat, top_k=24)
    return df.fillna('').reset_index(drop=True), (banner or '')


def check(name, fn):
    try:
        ok, info = fn()
    except Exception as e:          # a crash is a failure, not a stop
        ok, info = False, f'CRASH: {type(e).__name__}: {e}'
    results_log.append(ok)
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'   -> {info}'))


def t(df, n=5):
    return df['title'].tolist()[:n]


def a(df, n=5):
    return df['author'].tolist()[:n]


# --- names and typos ---
check('rowling (correct) -> Rowling books',
      lambda: ((lambda d, b: ('Rowling' in b and all('Rowling' in x for x in a(d, 24)), (b, a(d))))(*run('rowling'))))
check('roling (typo) -> Rowling books',
      lambda: ((lambda d, b: ('Rowling' in b, (b, t(d))))(*run('roling'))))
check('jk roling -> Rowling',
      lambda: ((lambda d, b: ('Rowling' in b, (b, t(d))))(*run('jk roling'))))
check('j k rolling -> Rowling',
      lambda: ((lambda d, b: ('Rowling' in b, (b, t(d))))(*run('j k rolling'))))
check('stephen king -> King books',
      lambda: ((lambda d, b: (all('king' in fold(x) for x in a(d)), (b, a(d))))(*run('stephen king'))))
check('kingg -> Stephen King shown',
      lambda: ((lambda d, b: ('Stephen King' in a(d, 24) or 'Stephen King' in b, (b, a(d))))(*run('kingg'))))
check('king -> no Mockingbird (whole-word match)',
      lambda: ((lambda d, b: ('To Kill a Mockingbird' not in t(d, 24), t(d, 24)))(*run('king'))))

# --- titles ---
check('harry potter -> Harry Potter first',
      lambda: ((lambda d, b: ('Harry Potter' in t(d, 1)[0], t(d)))(*run('harry potter'))))
check('harry poter -> corrected to Harry Potter',
      lambda: ((lambda d, b: ('Harry Potter' in t(d, 1)[0], (b, t(d))))(*run('harry poter'))))
HAS_HARRY_AUTHOR = wd['author_f'].str.contains(r'(?<!\w)harry(?!\w)', regex=True).any()
check('harry -> title section (and author section if authors named Harry exist)',
      lambda: ((lambda d, b: (d['section'].nunique() >= (2 if HAS_HARRY_AUTHOR else 1), d['section'].unique().tolist()))(*run('harry'))))
check('howling -> Howling books',
      lambda: ((lambda d, b: (any('howling' in x.lower() for x in t(d, 3)), t(d)))(*run('howling'))))
check('hobbit -> no author banner',
      lambda: ((lambda d, b: ('author' not in b.lower(), (b, t(d))))(*run('hobbit'))))
check('twilight -> no author banner',
      lambda: ((lambda d, b: ('author' not in b.lower(), (b, t(d))))(*run('twilight'))))

# --- accents ---
if wd['author_f'].str.contains('marquez').any():
    check('garcia marquez (no accents) -> Marquez books',
          lambda: ((lambda d, b: (any('marquez' in fold(x) for x in a(d, 8)), (b, a(d, 8))))(*run('garcia marquez'))))

# --- meaning / empty / categories ---
check('a book about loneliness -> results',
      lambda: ((lambda d, b: (len(d) >= 10, len(d)))(*run('a book about loneliness'))))
check('neural networks -> results',
      lambda: ((lambda d, b: (len(d) > 0, len(d)))(*run('neural networks'))))
check('empty search -> popular books',
      lambda: ((lambda d, b: (len(d) == 24, len(d)))(*run(''))))
check('empty + Computer Science -> only CS books',
      lambda: ((lambda d, b: ('Computer Science' in b and len(d) > 0 and all(
          any(k in c.lower() for k in ['computer', 'programming', 'software']) for c in d['categories']),
          (b, d['categories'].tolist()[:5])))(*run('', 'Computer Science'))))
check('gibberish does not crash',
      lambda: ((lambda d, b: (True, ''))(*run('asdfghjk'))))

print(f"\n{sum(results_log)}/{len(results_log)} checks passed")
