"""Shows which path an author-name typo takes through the search.
Run from D:/documents/Smart_Library:   python tests\\diagnose_author.py"""
import sys
sys.path.append('src')
import search as S
from search import load_search_system, proc, fold, process, fuzz, _search_core

S.live_lookup = lambda *a, **k: []                         # no internet during the test
S.get_better_cover = lambda isbn, title, author, fallback: fallback

s = load_search_system(sys.argv[1] if len(sys.argv) > 1 else 'data/processed')

for q in ['tolkein', 'tolkien', 'jane austin', 'jane austen']:
    p = proc(q)
    tb = process.extractOne(p, s['title_proc'], scorer=fuzz.token_sort_ratio)
    ab = process.extractOne(p, s['author_proc'], scorer=fuzz.token_sort_ratio)
    sm = process.extractOne(fold(q), s['surname_list'], scorer=fuzz.ratio, score_cutoff=85)
    df, banner = _search_core(q, s, top_k=5)
    print(repr(q))
    print('   best title                 :', tb and (s['title_list'][tb[2]], round(tb[1])))
    print('   best author                :', ab and (s['author_list'][ab[2]], round(ab[1])))
    print('   surname match              :', sm)
    print('   last word is in catalog    :', fold(q.split()[-1]) in s['vocab'])
    print('   banner                     :', banner)
    print('   top authors                :', df['author'].tolist()[:3])