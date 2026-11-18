"""Every way people name an author or a book: middle names, nicknames, pen names, real names, initials,
short forms, "sir", reversed order, punctuation - each also WITH typos - plus ISBNs in every format.
Run from D:/documents/Smart_Library_V2:   python tests/names.py
A search passes when one of the TOP 5 books matches the expected pattern (title or author),
or for ('say', text) when the page message contains that text."""
import re
import sys
import time
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import pandas as pd
import search as S
S.live_lookup = lambda *a, **k: []
S.get_better_cover = lambda isbn, title, author, fallback: fallback

CASES = [
    # middle names / full names
    ('Middle name', 'ronald tolkien', r'tolkien'),
    ('Middle name', 'john ronald reuel tolkien', r'tolkien'),
    ('Middle name', 'reuel tolkien', r'tolkien'),
    ('Middle name', 'kathleen rowling', r'rowling'),
    ('Middle name', 'joanne kathleen rowling', r'rowling'),
    ('Middle name', 'joanne rowling', r'rowling'),
    ('Middle name', 'arthur conan doyle', r'doyle'),
    ('Middle name', 'conan doyle', r'doyle'),
    ('Middle name', 'edgar allan poe', r'poe'),
    ('Middle name', 'edgar poe', r'poe'),
    ('Middle name', 'allan poe', r'poe'),
    ('Middle name', 'gabriel garcia marquez', r'marquez'),
    ('Middle name', 'garcia marquez', r'marquez'),
    ('Middle name', 'jorge luis borges', r'borges'),
    ('Middle name', 'luis borges', r'borges'),
    ('Middle name', 'francis scott fitzgerald', r'fitzgerald'),
    ('Middle name', 'scott fitzgerald', r'fitzgerald'),
    ('Middle name', 'george raymond richard martin', r'martin'),
    ('Middle name', 'ursula le guin', r'le guin'),
    ('Middle name', 'ursula kroeber le guin', r'le guin'),
    ('Middle name', 'howard phillips lovecraft', r'lovecraft'),
    # initials / nicknames / short forms of names
    ('Initials', 'jk rowling', r'rowling'),
    ('Initials', 'j.k. rowling', r'rowling'),
    ('Initials', 'j k rowling', r'rowling'),
    ('Initials', 'jrr tolkien', r'tolkien'),
    ('Initials', 'j.r.r. tolkien', r'tolkien'),
    ('Initials', 'hp lovecraft', r'lovecraft'),
    ('Initials', 'h.p. lovecraft', r'lovecraft'),
    ('Initials', 'grr martin', r'martin'),
    ('Initials', 'george rr martin', r'martin'),
    ('Initials', 'f scott fitzgerald', r'fitzgerald'),
    ('Initials', 'cs lewis', r'lewis'),
    ('Initials', 'c.s. lewis', r'lewis'),
    ('Initials', 'ts eliot', r'eliot'),
    ('Initials', 'eb white', r'white'),
    ('Initials', 'rl stine', r'stine'),
    ('Initials', 'jk', r'rowling'),
    ('Initials', 'jrr', r'tolkien'),
    ('Title', 'sir arthur conan doyle', r'doyle'),
    ('Name order', 'rowling jk', r'rowling'),
    ('Name order', 'tolkien jrr', r'tolkien'),
    ('Name order', 'king stephen', r'stephen king'),
    ('Name order', 'austen, jane', r'austen'),
    ('Name order', 'doyle arthur conan', r'doyle'),
    # pen names <-> real names
    ('Pen / real name', 'mark twain', r'twain|clemens'),
    ('Pen / real name', 'samuel clemens', r'twain|clemens'),
    ('Pen / real name', 'samuel langhorne clemens', r'twain|clemens'),
    ('Pen / real name', 'george orwell', r'orwell'),
    ('Pen / real name', 'eric blair', r'orwell'),
    ('Pen / real name', 'eric arthur blair', r'orwell'),
    ('Pen / real name', 'lewis carroll', r'carroll|dodgson'),
    ('Pen / real name', 'charles dodgson', r'carroll|dodgson'),
    ('Pen / real name', 'dr seuss', r'seuss|geisel'),
    ('Pen / real name', 'dr. seuss', r'seuss|geisel'),
    ('Pen / real name', 'theodor geisel', r'seuss|geisel'),
    ('Pen / real name', 'robert galbraith', r'galbraith|rowling'),
    ('Pen / real name', 'richard bachman', r'bachman|stephen king'),
    ('Pen / real name', 'mary westmacott', r'westmacott|christie'),
    ('Pen / real name', 'currer bell', r'bronte|bell'),
    ('Pen / real name', 'george eliot', r'eliot'),
    ('Pen / real name', 'mary ann evans', r'eliot|evans'),
    ('Pen / real name', 'voltaire', r'voltaire'),
    ('Pen / real name', 'lemony snicket', r'snicket|handler'),
    # common names / well-known short names
    ('Common name', 'shakespeare', r'shakespeare'),
    ('Common name', 'dickens', r'dickens'),
    ('Common name', 'tolstoy', r'tolstoy|tolstoi'),
    ('Common name', 'dostoevsky', r'dostoevsk|dostoyevsk'),
    ('Common name', 'dostoyevsky', r'dostoevsk|dostoyevsk'),
    ('Common name', 'chekhov', r'chekhov|tchekhov|chekov'),
    ('Common name', 'homer', r'homer'),
    ('Common name', 'plato', r'plato'),
    ('Common name', 'kafka', r'kafka'),
    ('Common name', 'seuss', r'seuss'),
    # the same, WITH typos
    ('Name + typo', 'ronald tolkein', r'tolkien'),
    ('Name + typo', 'kathleen rowlig', r'rowling'),
    ('Name + typo', 'arthur conan doyel', r'doyle'),
    ('Name + typo', 'conan doile', r'doyle'),
    ('Name + typo', 'edgar alan poe', r'poe'),
    ('Name + typo', 'edgar allen poe', r'poe'),
    ('Name + typo', 'garcia marques', r'marquez'),
    ('Name + typo', 'gabriel garcia markez', r'marquez'),
    ('Name + typo', 'jorge luis borjes', r'borges'),
    ('Name + typo', 'scott fitzgerlad', r'fitzgerald'),
    ('Name + typo', 'hp lovecraf', r'lovecraft'),
    ('Name + typo', 'h p lovecrft', r'lovecraft'),
    ('Name + typo', 'george rr martn', r'martin'),
    ('Name + typo', 'cs lewiss', r'lewis'),
    ('Name + typo', 'mark twian', r'twain'),
    ('Name + typo', 'samuel clemmens', r'twain|clemens'),
    ('Name + typo', 'lewis caroll', r'carroll'),
    ('Name + typo', 'dr suess', r'seuss'),
    ('Name + typo', 'shakespear', r'shakespeare'),
    ('Name + typo', 'shakespere', r'shakespeare'),
    ('Name + typo', 'dikens', r'dickens'),
    ('Name + typo', 'tolstoj', r'tolstoy|tolstoi'),
    ('Name + typo', 'kafak', r'kafka'),
    # short forms of titles / subjects
    ('Short form', 'lotr', r'lord of the rings'),
    ('Short form', 'hhgttg', r'hitchhiker'),
    ('Short form', 'ml', r'machine learning'),
    ('Short form', 'ai', r'artificial intelligence'),
    ('Short form', 'dsa', r'data structures|algorithm'),
    ('Short form', 'dbms', r'database'),
    ('Short form', 'oop', r'object'),
    ('Short form', 'os', r'operating system'),
    ('Short form', 'nlp', r'natural language'),
    ('Short form', 'hp and the goblet of fire', r'goblet of fire'),
    ('Short form', 'lotr return of the king', r'return of the king'),
    # ISBN in every format
    ('ISBN', '9780590353427', r'sorcerer'),
    ('ISBN', '978-0-590-35342-7', r'sorcerer'),
    ('ISBN', '978 0590 353427', r'sorcerer'),
    ('ISBN', '0590353403', r'sorcerer|harry potter'),
    ('ISBN', 'isbn 9780590353427', r'sorcerer'),
    ('ISBN', 'ISBN: 978-0-590-35342-7', r'sorcerer'),
    ('ISBN', 'isbn9780590353427', r'sorcerer'),
    ('ISBN', '9780345339683', r'hobbit'),
    ('ISBN', '9780451519849', r'1984'),
    ('ISBN', '9780684801520', r'gatsby'),
    ('ISBN typo', '9780590353428', ('say', 'isbn')),        # one digit wrong: must say so, not show random books
    ('ISBN typo', '978059035342', ('say', 'isbn')),         # one digit missing
    ('ISBN', 'isbn 0590353403', r'sorcerer|harry potter'),
    ('ISBN', 'ISBN-13: 9780590353427', r'sorcerer'),
    ('ISBN', 'isbn-10 0590353403', r'sorcerer|harry potter'),
    ('ISBN', 'isbn of harry potter', r'harry potter'),
    ('Pen / real name', 'eric blair', ('say', 'george orwell')),
]


def top_text(df, n=5):
    rows = df.head(n).fillna('')
    return [S.fold(f"{r.get('title', '')} | {r.get('author', '')}") for _, r in rows.iterrows()]


if __name__ == '__main__':
    system = S.load_search_system('data/processed')
    rows = []
    for kind, q, expect in CASES:
        t = time.time()
        df, banner = S.search_books(q, system, top_k=24)
        sec = time.time() - t
        top = top_text(df) if len(df) else []
        if isinstance(expect, tuple):
            ok = expect[1] in (banner or '').lower()
        else:
            ok = len(df) > 0 and any(re.search(expect, x) for x in top)
        rows.append({'kind': kind, 'query': q, 'ok': ok, 'sec': round(sec, 2), 'banner': (banner or '')[:90],
                     'top': ' || '.join(x[:60] for x in top[:3])})
    res = pd.DataFrame(rows)
    res.to_csv('tests/names.csv', index=False)
    print(f"{res['ok'].sum()}/{len(res)} name / short-form / ISBN searches found the right books")
    print(f"slowest: {res['sec'].max()} s ({res.loc[res['sec'].idxmax(), 'query']!r})\n")
    for kind, r in res.groupby('kind', sort=False)['ok'].agg(['sum', 'count']).iterrows():
        print(f"  {'OK  ' if r['sum'] == r['count'] else 'FAIL'} {kind:18} {r['sum']}/{r['count']}")
    print('\n=== need fixing ===')
    for _, r in res[~res['ok']].iterrows():
        print(f"  [{r['kind']}] {r['query']!r}\n      message: {r['banner'] or '-'}\n      top: {r['top'] or '(nothing)'}")
