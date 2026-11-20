"""Broad typo sweep: takes REAL authors and titles out of the catalog (famous and random, fiction and
textbooks, every source) and searches them with many kinds of mistakes: a dropped, doubled, swapped or
wrong-key letter, a same-sound spelling (podder/potter), a half-typed word, initials, surname only.

A search passes when the book/author we started from shows up in the TOP 10 results.
Run from D:/documents/Smart_Library_V2:
    python tests/typo_sweep.py                 # about 20-30 minutes, writes tests/typo_sweep.csv
    python tests/typo_sweep.py --quick         # about 5 minutes (smaller sample)
    python tests/typo_sweep.py --only-file tests/typo_sweep_failed.txt   # re-run just the earlier failures
Set SRC_DIR=some/other/src to compare an older copy of search.py."""
import os
import random
import re
import sys
import time
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.environ.get('SRC_DIR', 'src'))
import pandas as pd
import search as S
from rapidfuzz import fuzz

S.live_lookup = lambda *a, **k: []
S.get_better_cover = lambda isbn, title, author, fallback: fallback

QUICK = '--quick' in sys.argv
SEED = 7
rng = random.Random(SEED)
KEYS = {c: n for c, n in zip('qwertyuiopasdfghjklzxcvbnm', [
    'wa', 'qes', 'wrd', 'etf', 'rgy', 'tuh', 'yij', 'uok', 'ilp', 'op', 'qwsz', 'wedxza', 'erfcxs', 'rtgvcd',
    'tyhbvf', 'yujnbg', 'uikmnh', 'iolmj', 'opk', 'asx', 'sdc', 'dfvx', 'fgbc', 'ghnv', 'hjmb', 'jkn'])}
SOUND = [('dd', 'tt'), ('tt', 'dd'), ('bb', 'pp'), ('ph', 'f'), ('f', 'ph'), ('ck', 'k'), ('c', 'k'), ('k', 'c'),
         ('ie', 'ei'), ('ei', 'ie'), ('ee', 'ea'), ('ea', 'ee'), ('s', 'z'), ('y', 'i'), ('i', 'y'), ('ou', 'u'),
         ('oo', 'u'), ('ght', 't'), ('tion', 'shun'), ('ch', 'sh')]


def words_of(text):
    return [w for w in re.split(r'\s+', S.fold(text).strip()) if w]


def mut_drop(ws):
    idx = [i for i, w in enumerate(ws) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(ws[i]) - 1)
    return ws[:i] + [ws[i][:j] + ws[i][j + 1:]] + ws[i + 1:]


def mut_double(ws):
    idx = [i for i, w in enumerate(ws) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(ws[i]))
    return ws[:i] + [ws[i][:j] + ws[i][j] + ws[i][j:]] + ws[i + 1:]


def mut_swap(ws):
    idx = [i for i, w in enumerate(ws) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(ws[i]) - 2)
    w = ws[i]
    return ws[:i] + [w[:j] + w[j + 1] + w[j] + w[j + 2:]] + ws[i + 1:]


def mut_key(ws):
    idx = [i for i, w in enumerate(ws) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return None
    i = rng.choice(idx)
    j = rng.randrange(1, len(ws[i]))
    near = KEYS.get(ws[i][j])
    if not near:
        return None
    return ws[:i] + [ws[i][:j] + rng.choice(near) + ws[i][j + 1:]] + ws[i + 1:]


def mut_sound(ws):
    cands = [(i, a, b) for i, w in enumerate(ws) if len(w) >= 5 and w.isalpha() for a, b in SOUND if a in w[1:]]
    if not cands:
        return None
    i, a, b = rng.choice(cands)
    w = ws[i]
    k = w.index(a, 1)
    return ws[:i] + [w[:k] + b + w[k + len(a):]] + ws[i + 1:]


def mut_half(ws):
    i = len(ws) - 1
    if len(ws[i]) < 6 or not ws[i].isalpha():
        return None
    return ws[:i] + [ws[i][:rng.choice([3, 4, len(ws[i]) - 2])]]


def mut_two(ws):
    a = mut_drop(ws) or mut_double(ws)
    return mut_swap(a) if a else None


MUTATIONS = {'dropped letter': mut_drop, 'doubled letter': mut_double, 'swapped letters': mut_swap,
             'wrong key': mut_key, 'sounds the same': mut_sound, 'half typed': mut_half, 'two typos': mut_two}


def clean_title(t):
    t = re.split(r'[:(\[]', t)[0].strip()
    return S.fold(t)


def main():
    system = S.load_search_system('data/processed')
    wd = system['work_dedup'].copy()
    wd['rc'] = pd.to_numeric(wd['rating_count'], errors='coerce').fillna(0)
    n_auth, n_famous_t, n_src_t = (40, 60, 8) if QUICK else (160, 250, 25)

    # ---- authors: the best-known, plus a random pick of the rest (3+ books) ----
    fame = system['author_fame']
    ranked = sorted(fame, key=lambda a: -fame[a])
    pool = [a for a in ranked if re.fullmatch(r"[a-z .'-]+", S.fold(system['author_display'].get(a, a)))]
    top = pool[:n_auth * 2]
    rest = rng.sample(pool[n_auth * 2:], min(len(pool) - n_auth * 2, n_auth * 2))
    authors = rng.sample(top, min(n_auth, len(top))) + rng.sample(rest, min(n_auth, len(rest)))

    # ---- titles: famous, random, and every source (textbooks, programming, Gutenberg, ...) ----
    titles = []
    famous = wd.sort_values('rc', ascending=False).head(3000)
    for _, r in famous.sample(n_famous_t, random_state=SEED).iterrows():
        titles.append(('famous', r['title'], r['author']))
    for src, grp in wd.groupby('source'):
        grp = grp[grp['title'].str.split().str.len() >= 2]
        for _, r in grp.sample(min(n_src_t, len(grp)), random_state=SEED).iterrows():
            titles.append((f'source {src}', r['title'], r['author']))

    queries = []                      # (group, mutation, typed query, kind, expected)
    for a in authors:
        disp = system['author_display'].get(a, a)
        ws = words_of(disp)
        ws = [w.strip('.') for w in ws if len(w.strip('.')) > 1] or ws
        if len(ws) < 2 or len(ws[-1]) < 4:
            continue
        sur = ws[-1]
        queries.append(('author', 'surname only', sur, 'author', sur))
        queries.append(('author', 'first + last', ' '.join(ws[:1] + ws[-1:]), 'author', sur))
        queries.append(('author', 'initial + last', ws[0][0] + ' ' + sur, 'author', sur))
        for name, fn in MUTATIONS.items():
            m = fn(ws[:1] + ws[-1:])
            if m and ' '.join(m) != ' '.join(ws[:1] + ws[-1:]):
                queries.append(('author', name, ' '.join(m), 'author', sur))
        m = mut_drop([sur]) or mut_swap([sur])
        if m:
            queries.append(('author', 'surname typo only', m[0], 'author', sur))
    for grp, title, author in titles:
        base = clean_title(title)
        ws = words_of(base)
        if len(ws) < 2 or len(base) > 60:
            continue
        queries.append((grp, 'correct title', base, 'title', base))
        for name, fn in MUTATIONS.items():
            m = fn(ws)
            if m and ' '.join(m) != base:
                queries.append((grp, name, ' '.join(m), 'title', base))

    only = None
    if '--only-file' in sys.argv:
        only = set(open(sys.argv[sys.argv.index('--only-file') + 1], encoding='utf-8').read().split('\n'))
        queries = [q for q in queries if q[2] in only]
    print(f'{len(queries)} searches ({len(authors)} authors, {len(titles)} titles)')

    rows = []
    t0 = time.time()
    for n, (grp, mutation, q, kind, expected) in enumerate(queries, 1):
        try:
            df, banner = S.search_books(q, system, top_k=10)
            df = df.fillna('').head(10)
            if kind == 'author':
                ok = any(expected in S.fold(a) for a in df['author'])
            else:
                exp = expected[:40]
                ok = any(fuzz.ratio(exp, clean_title(t)[:40]) >= 80 for t in df['title'])
            top1 = (df['title'].iloc[0] + ' / ' + df['author'].iloc[0]) if len(df) else ''
        except Exception as e:                    # a crash is a failure
            ok, banner, top1 = False, f'CRASH {type(e).__name__}: {e}', ''
        rows.append(dict(group=grp, mutation=mutation, query=q, expected=expected, ok=ok,
                         banner=str(banner or '')[:70], top=top1[:70]))
        if n % 100 == 0:
            print(f'  {n}/{len(queries)}  {time.time() - t0:.0f}s  so far {100 * sum(r["ok"] for r in rows) / len(rows):.0f}%', flush=True)

    res = pd.DataFrame(rows)
    res.to_csv('tests/typo_sweep.csv', index=False)
    print('\n=== pass rate by kind of mistake ===')
    print((res.groupby('mutation')['ok'].mean() * 100).round(0).astype(int).astype(str).add('%').to_string())
    print('\n=== pass rate by group ===')
    print((res.groupby('group')['ok'].mean() * 100).round(0).astype(int).astype(str).add('%').to_string())
    bad = res[~res['ok']]
    print(f'\nOVERALL {100 * res["ok"].mean():.1f}%   failures: {len(bad)}   crashes: {int(res["banner"].str.startswith("CRASH").sum())}')
    with open('tests/typo_sweep_failed.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(bad['query']))


if __name__ == '__main__':
    main()
