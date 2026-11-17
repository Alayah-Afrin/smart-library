"""Try every query x category x free-only combination and flag confusing results.
Run from D:/documents/Smart_Library_V2:   python tests/filter_matrix.py
A result list is 'confusing' when it has books but none of the top 5 mention any searched word
(in title, author, category or description) and the banner gives no explanation.
Writes the full table to tests/filter_matrix.csv."""
import re
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import pandas as pd
import search as S
S.live_lookup = lambda *a, **k: []
S.get_better_cover = lambda isbn, title, author, fallback: fallback
from search import load_search_system, search_books, fold, strip_stopwords, CATEGORY_GROUPS

system = load_search_system('data/processed')
QUERIES = ['harry', 'harry potter', 'rowling', 'roling', 'tolkien', 'stephen king', 'king', 'jane austen',
           'charles dickens', 'calculus', 'python programming', 'linear algebra', 'neural networks',
           'a book about loneliness', 'sherlock holmes', '1984', 'love', 'war', 'asdfghjk', '',
           # typos, author phrases, long descriptions, unicode, numbers, single letters
           'calculas', 'phyton', 'shakspeare', 'books by dickens', 'written by mark twain', 'garcía márquez',
           'machine learning for beginners', 'a funny adventure for kids', 'free physics textbook',
           'romeo and juliet', 'pride and prejudice', 'organic chemistry', 'statistics', 'cooking',
           'history of rome', '2001', 'a', 'the', 'c++', 'data structures and algorithms']
CATS = [None] + list(CATEGORY_GROUPS)
EXPLAINS = ('No ', 'Did you mean', 'Popular in', 'Free to read', 'Searched for', 'Showing', 'nothing', 'Try ')


def words(q):
    return [w for w in strip_stopwords(fold(q)).split() if len(w) > 2]


rows = []
for q in QUERIES:
    for cat in CATS:
        for free in (False, True):
            try:
                df, banner = search_books(q, system, category_filter=cat, top_k=24, free_only=free)
                err = ''
            except Exception as e:
                df, banner, err = pd.DataFrame(), '', f'{type(e).__name__}: {e}'
            banner = banner or ''
            top = df.head(5).fillna('') if len(df) else df
            text = [fold(' '.join(str(r.get(c, '')) for c in ('title', 'author', 'categories', 'description')))
                    for _, r in top.iterrows()]
            ws = words(q)
            hits = sum(any(re.search(r'(?<!\w)' + re.escape(w), t) for w in ws) for t in text) if ws else len(text)
            confusing = bool(err) or bool(len(df) > 0 and ws and hits == 0 and not banner.startswith(EXPLAINS))
            cat_ok = True
            if cat and len(df) and 'categories' in df:
                pat = '|'.join(CATEGORY_GROUPS[cat])
                cat_ok = bool(df['categories'].fillna('').str.contains(pat, case=False).all())
            free_ok = (not free) or len(df) == 0 or bool((df.get('read_url', pd.Series([''])).fillna('') != '').all())
            sections = ' / '.join(s for s in df['section'].fillna('').unique() if s) if 'section' in df else ''
            rows.append({'query': q, 'category': cat or '-', 'free': free, 'n': len(df), 'banner': banner[:90],
                         'sections': sections,
                         'top3': ' | '.join(str(t)[:35] for t in df['title'].head(3)) if len(df) else '',
                         'word_hits_top5': hits, 'confusing': confusing, 'category_respected': cat_ok,
                         'free_respected': free_ok, 'error': err})

res = pd.DataFrame(rows)
res.to_csv('tests/filter_matrix.csv', index=False)
print(f"{len(res)} combinations tried.")
print(f"  crashes: {(res['error'] != '').sum()}")
print(f"  category ignored: {(~res['category_respected']).sum()}")
print(f"  free filter ignored: {(~res['free_respected']).sum()}")
print(f"  confusing (unrelated books, no explanation): {res['confusing'].sum()}")
print(f"  empty results: {(res['n'] == 0).sum()}")
print("\n=== 'harry' with each category (what a user would see) ===")
h = res[res['query'] == 'harry']
print(h[['category', 'free', 'n', 'banner', 'sections', 'top3']].to_string(index=False, max_colwidth=70))
bad = res[res['confusing'] | ~res['category_respected'] | ~res['free_respected'] | (res['error'] != '')]
if len(bad):
    print("\n=== PROBLEMS ===")
    print(bad[['query', 'category', 'free', 'n', 'banner', 'top3', 'error']].to_string(index=False, max_colwidth=60))
