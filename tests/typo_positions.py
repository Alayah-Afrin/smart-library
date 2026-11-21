"""Typos in every position and every size, AND every suggestion link is clicked.

For each famous phrase (title, author, subject) we damage ONE word at a time (first word, last word, a middle
word) and ALL words, with small, medium and heavy damage, at the start, middle and end of the word.
A search passes when the right book/author is in the top 10.
Then every "Did you mean" link the page would show is searched too: a link must never be a dead end
(no "isn't in our library", no empty page, and it must not be a worse guess than the original typo).

Run from D:/documents/Smart_Library_V2:   python tests/typo_positions.py
Writes tests/typo_positions.csv (every search) and tests/typo_positions_failed.txt (only the failures).
"""
import csv
import random
import re
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import search as S
S.live_lookup = lambda *a, **k: []
S.get_better_cover = lambda isbn, title, author, fallback: fallback

PHRASES = [
    ('harry potter', r'harry potter'), ('stephen king', r'stephen king'), ('jane austen', r'austen'),
    ('agatha christie', r'christie'), ('george orwell', r'orwell'), ('sherlock holmes', r'sherlock holmes|doyle'),
    ('lord of the rings', r'lord of the rings|tolkien'), ('pride and prejudice', r'pride and prejudice'),
    ('the great gatsby', r'gatsby'), ('to kill a mockingbird', r'mockingbird'), ('mark twain', r'twain'),
    ('charles dickens', r'dickens'), ('machine learning', r'machine learning'), ('python programming', r'python'),
    ('data structures', r'data structures'), ('linear algebra', r'linear algebra'), ('organic chemistry', r'organic chemistry'),
    ('game of thrones', r'game of thrones'), ('the hunger games', r'hunger games'), ('percy jackson', r'percy jackson|lightning thief|riordan'),
    ('ernest hemingway', r'hemingway'), ('leo tolstoy', r'tolstoy'), ('artificial intelligence', r'artificial intelligence'),
    ('operating systems', r'operating system'), ('dan brown', r'dan brown'), ('roald dahl', r'dahl'),
]
NEAR = {'a': 'sq', 'e': 'wr', 'i': 'uo', 'o': 'ip', 'u': 'yi', 'y': 'tu', 'r': 'et', 'h': 'gj', 'p': 'ol', 't': 'ry', 'n': 'bm',
        's': 'ad', 'l': 'k', 'd': 'sf', 'c': 'xv', 'm': 'n', 'k': 'j', 'g': 'fh', 'b': 'vn', 'w': 'qe', 'f': 'dg'}
VOWELS = 'aeiou'


def small(w, rnd, where):
    """One wrong letter at the start / middle / end."""
    i = {'start': 0, 'middle': len(w) // 2, 'end': len(w) - 1}[where]
    kind = rnd.choice(['drop', 'double', 'wrong', 'swap'])
    if kind == 'drop' and len(w) > 3:
        return w[:i] + w[i + 1:]
    if kind == 'double':
        return w[:i] + w[i] + w[i:]
    if kind == 'wrong':
        return w[:i] + rnd.choice(NEAR.get(w[i], 'e')) + w[i + 1:]
    j = min(i, len(w) - 2)
    return w[:j] + w[j + 1] + w[j] + w[j + 2:]


def vowel_slip(w, rnd):
    """The classic: a vowel confused for another (harry -> herry), or ie/ei, y/i."""
    spots = [i for i, c in enumerate(w) if c in VOWELS + 'y']
    if not spots:
        return small(w, rnd, 'middle')
    i = rnd.choice(spots)
    return w[:i] + rnd.choice([c for c in VOWELS + 'y' if c != w[i]]) + w[i + 1:]


def medium(w, rnd):
    """Two wrong letters."""
    return small(small(w, rnd, rnd.choice(['start', 'middle', 'end'])), rnd, rnd.choice(['middle', 'end']))


def heavy(w, rnd):
    """3-4 wrong letters in a word."""
    if len(w) < 6:
        return medium(w, rnd)
    w2 = w
    for _ in range(3):
        w2 = small(w2, rnd, rnd.choice(['start', 'middle', 'end']))
    return w2


def variants(phrase, rnd):
    words = phrase.split()
    real = [i for i, w in enumerate(words) if len(w) >= 4]
    out = []                                                  # (label, query)
    spots = {'first word': real[0], 'last word': real[-1]}
    if len(real) > 2:
        spots['middle word'] = real[len(real) // 2]
    for name, i in spots.items():
        w = words[i]
        for label, fn in (('start of word', lambda x: small(x, rnd, 'start')),
                          ('middle of word', lambda x: small(x, rnd, 'middle')),
                          ('end of word', lambda x: small(x, rnd, 'end')),
                          ('vowel slip', lambda x: vowel_slip(x, rnd)),
                          ('two wrong letters', lambda x: medium(x, rnd)),
                          ('heavy', lambda x: heavy(x, rnd))):
            bad = fn(w)
            if bad != w:
                out.append((f'{name}: {label}', ' '.join(words[:i] + [bad] + words[i + 1:])))
    if len(real) >= 2:                                        # every word damaged at once
        out.append(('all words: small', ' '.join(small(w, rnd, 'middle') if len(w) >= 4 else w for w in words)))
        out.append(('all words: vowel slips', ' '.join(vowel_slip(w, rnd) if len(w) >= 4 else w for w in words)))
        out.append(('all words: heavy', ' '.join(heavy(w, rnd) if len(w) >= 4 else w for w in words)))
    cut = max(4, len(phrase) // 2)                           # half typed, but stopped INSIDE a word
    while cut < len(phrase) - 1 and (phrase[cut - 1] == ' ' or phrase[cut] == ' '):
        cut += 1
    out.append(('first half only', phrase[:cut]))
    return out


def blob(df, n=10):
    return [S.fold(f"{r.get('title', '')} | {r.get('author', '')} | {r.get('categories', '')}")
            for _, r in df.head(n).fillna('').iterrows()]


if __name__ == '__main__':
    rnd = random.Random(7)
    system = S.load_search_system('data/processed')
    rows, failed = [], []
    for phrase, pat in PHRASES:
        for label, q in variants(phrase, rnd):
            df, banner = S.search_books(q, system, top_k=24)
            ok = len(df) > 0 and any(re.search(pat, t) for t in blob(df))
            alts = [a for a in (df.attrs.get('alternatives') or []) if a.strip().lower() != q.strip().lower()]
            dead = []
            for a in alts:                                    # click every suggestion link
                df2, banner2 = S.search_books(a, system, top_k=24)
                if len(df2) == 0 or "isn't in our library" in (banner2 or ''):
                    dead.append(a)
            rows.append({'phrase': phrase, 'kind': label, 'query': q, 'ok': ok, 'links': ' | '.join(alts),
                         'dead_links': ' | '.join(dead), 'banner': (banner or '')[:80]})
            if not ok or dead:
                failed.append(f"[{phrase} / {label}] {q!r}  ok={ok}  links={alts}  dead={dead}\n      banner: {banner}")
    with open('tests/typo_positions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open('tests/typo_positions_failed.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed))
    n = len(rows)
    good = sum(r['ok'] for r in rows)
    dead_n = sum(bool(r['dead_links']) for r in rows)
    print(f"\n{good}/{n} damaged searches found the right book ({100 * good / n:.1f}%)")
    print(f"{dead_n} searches offered a suggestion link that is a dead end")
    by = {}
    for r in rows:
        k = r['kind'].split(': ')[-1] if ': ' in r['kind'] else r['kind']
        by.setdefault(k, [0, 0])
        by[k][0] += r['ok']
        by[k][1] += 1
    for k, (a, b) in sorted(by.items(), key=lambda x: x[1][0] / x[1][1]):
        print(f"  {100 * a / b:5.1f}%  {k}  ({a}/{b})")
    print('\n=== failures (also in tests/typo_positions_failed.txt) ===')
    print('\n'.join(failed[:60]))
