"""Spelling mistakes of every kind: author names, titles, subjects, categories, possessives, extra letters,
missing letters, swapped letters, wrong keys, and mixes of a good word with a bad one.
Run from D:/documents/Smart_Library_V2:   python tests/typos.py
A search passes when one of the TOP 5 books matches the expected pattern (title, author or category)."""
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

TYPOS = [
    # author: initials + misspelled surname
    ('Author typo', 'jk rolimg', r'rowling'),
    ('Author typo', 'jk rowlng', r'rowling'),
    ('Author typo', 'j k rowlin', r'rowling'),
    ('Author typo', 'j.k. rowlling', r'rowling'),
    ('Author typo', 'jrr tolkein', r'tolkien'),
    ('Author typo', 'j r r tolkin', r'tolkien'),
    # author: full name, one or both words wrong
    ('Author typo', 'stephen kng', r'stephen king'),
    ('Author typo', 'stehpen king', r'stephen king'),
    ('Author typo', 'stephan kimg', r'stephen king'),
    ('Author typo', 'agatha cristie', r'christie'),
    ('Author typo', 'agata christie', r'christie'),
    ('Author typo', 'jane austin', r'austen'),
    ('Author typo', 'jane austne', r'austen'),
    ('Author typo', 'charles dikens', r'dickens'),
    ('Author typo', 'charls dickens', r'dickens'),
    ('Author typo', 'mark twian', r'twain'),
    ('Author typo', 'george orwel', r'orwell'),
    ('Author typo', 'goerge orwell', r'orwell'),
    ('Author typo', 'william shakespere', r'shakespeare'),
    ('Author typo', 'ernest hemingway', r'hemingway'),
    ('Author typo', 'ernest hemmingway', r'hemingway'),
    ('Author typo', 'dan brwon', r'dan brown'),
    ('Author typo', 'paulo coehlo', r'coelho'),
    ('Author typo', 'leo tolstoi', r'tolstoy'),
    ('Author typo', 'fyodor dostoyevski', r'dostoevsky|dostoyevsky'),
    ('Author typo', 'haruki murakmi', r'murakami'),
    ('Author typo', 'neil gaimen', r'gaiman'),
    ('Author typo', 'rick riordon', r'riordan'),
    ('Author typo', 'suzanne colins', r'collins'),
    ('Author typo', 'john grishem', r'grisham'),
    ('Author typo', 'dr seus', r'seuss'),
    ('Author typo', 'roald dhal', r'dahl'),
    # author: one word
    ('Author one word typo', 'rowlng', r'rowling'),
    ('Author one word typo', 'tolkein', r'tolkien'),
    ('Author one word typo', 'shakspeare', r'shakespeare'),
    ('Author one word typo', 'hemmingway', r'hemingway'),
    ('Author one word typo', 'dostoevski', r'dostoevsky|dostoyevsky'),
    ('Author one word typo', 'christe', r'christie'),
    ('Author one word typo', 'grisham', r'grisham'),
    ('Author one word typo', 'rolimg', r'rowling'),
    # one misspelled word that is BOTH a title word and a name: the much-read meaning must be there
    ('Two meanings typo', 'sherlok', r'sherlock holmes|doyle'),
    ('Two meanings typo', 'serlock', r'sherlock holmes|doyle'),
    ('Two meanings typo', 'sherlck', r'sherlock holmes|doyle'),
    ('Two meanings typo', 'potterr', r'harry potter'),
    ('Two meanings typo', 'hobbitt', r'hobbit'),
    ('Two meanings typo', 'twilght', r'twilight'),
    # series known by the hero's name; first letter missing or wrong
    ('Series typo', 'percy jackson', r'lightning thief|sea of monsters|titan|labyrinth|olympian|percy jackson'),
    ('Series typo', 'orcy jackso', r'lightning thief|sea of monsters|titan|labyrinth|olympian|percy jackson'),
    ('Series typo', 'ercy jackson', r'lightning thief|sea of monsters|titan|labyrinth|olympian|percy jackson'),
    ('Series typo', 'percy jakson', r'lightning thief|sea of monsters|titan|labyrinth|olympian|percy jackson'),
    # famous characters (book data rarely names them): the series must come first
    ('Character', 'hermione', r'harry potter'),
    ('Character', 'hermoine', r'harry potter'),
    ('Character', 'dumbledore', r'harry potter'),
    ('Character', 'gandalf', r'lord of the rings|hobbit'),
    ('Character', 'frodo', r'lord of the rings|hobbit'),
    ('Character', 'katniss', r'hunger games'),
    ('Character', 'poirot', r'christie|poirot|orient express|roger ackroyd'),
    ('Topic word', 'a book about loneliness', r'loneliness'),
    # titles: one word wrong
    ('Title typo', 'harry poter', r'harry potter'),
    ('Title typo', 'hary potter', r'harry potter'),
    ('Title typo', 'harry potter and the philosphers stone', r'harry potter'),
    ('Title typo', 'harry potter and the chamber of secrests', r'chamber of secrets'),
    ('Title typo', 'the grate gatsby', r'gatsby'),
    ('Title typo', 'the great gatbsy', r'gatsby'),
    ('Title typo', 'to kill a mockinbird', r'mockingbird'),
    ('Title typo', 'pride and predjudice', r'pride and prejudice'),
    ('Title typo', 'lord of the rigns', r'lord of the rings'),
    ('Title typo', 'the hobbitt', r'hobbit'),
    ('Title typo', 'hunger gmaes', r'hunger games'),
    ('Title typo', 'catcher in the rye', r'catcher in the rye'),
    ('Title typo', 'cather in the rye', r'catcher in the rye'),
    ('Title typo', 'the da vinci cod', r'da vinci code'),
    ('Title typo', 'da vinchi code', r'da vinci code'),
    ('Title typo', 'twilite', r'twilight'),
    ('Title typo', 'frankenstine', r'frankenstein'),
    ('Title typo', 'dracual', r'dracula'),
    ('Title typo', 'the alchmist', r'alchemist'),
    ('Title typo', 'nineteen eighty for', r'nineteen eighty|1984'),
    ('Title typo', 'animal farn', r'animal farm'),
    ('Title typo', 'the lion the witch and the wardobe', r'wardrobe'),
    ('Title typo', 'little woman', r'little wom[ae]n'),
    ('Title typo', 'wuthering hieghts', r'wuthering heights'),
    ('Title typo', 'jane eyer', r'jane eyre'),
    ('Title typo', 'the kite runer', r'kite runner'),
    ('Title typo', 'gone with the wnd', r'gone with the wind'),
    ('Title typo', 'percy jakson', ('say', 'percy jackson')),   # no Percy Jackson book in the catalog: the message must fix the spelling
    ('Title typo', 'game of thrones', r'game of thrones'),
    ('Title typo', 'game of throns', r'game of thrones'),
    # subjects / categories / study words
    ('Subject typo', "chilfrend's book", r"child|juvenile"),
    ('Subject typo', "childrens books", r"child|juvenile"),
    ('Subject typo', "childern books", r"child|juvenile"),
    ('Subject typo', 'chldren stories', r"child|juvenile"),
    ('Subject typo', 'mathmatics', r'mathemat'),
    ('Subject typo', 'calculas', r'calculus'),
    ('Subject typo', 'linear algebar', r'linear algebra'),
    ('Subject typo', 'phsyics', r'physics'),
    ('Subject typo', 'chemestry', r'chemistry'),
    ('Subject typo', 'biolgy', r'biology'),
    ('Subject typo', 'programing in pyton', r'python'),
    ('Subject typo', 'pyhton programming', r'python'),
    ('Subject typo', 'machine lerning', r'machine learning'),
    ('Subject typo', 'artifical intelligence', r'artificial intelligence'),
    ('Subject typo', 'data strucures', r'data structures'),
    ('Subject typo', 'algoritms', r'algorithm'),
    ('Subject typo', 'econmics', r'economic'),
    ('Subject typo', 'psycology', r'psycholog'),
    ('Subject typo', 'philosphy', r'philosoph'),
    ('Subject typo', 'histroy of rome', r'rome|roman'),
    ('Subject typo', 'scince fiction', r'science fiction|sci'),
    ('Subject typo', 'fantacy novels', r'fantas'),
    ('Subject typo', 'detectiv stories', r'detective|mystery|sherlock'),
    ('Subject typo', 'cookng recipes', r'cook|recipe'),
    # keyboard / doubled / missing letters in a sentence
    ('Sentence typo', 'books abuot dragons', r'dragon'),
    ('Sentence typo', 'a book about lonelyness', r'lonel'),
    ('Sentence typo', 'storys about freindship', r'friend'),
    ('Sentence typo', 'wizzard school', r'wizard|harry potter|magic'),
    ('Sentence typo', 'vampire romanse', r'vampire|twilight'),
]


def top_text(df, n=5):
    rows = df.head(n).fillna('')
    return [S.fold(f"{r.get('title', '')} | {r.get('author', '')} | {r.get('categories', '')}") for _, r in rows.iterrows()]


if __name__ == '__main__':
    system = S.load_search_system('data/processed')
    rows = []
    for kind, q, expect in TYPOS:
        t = time.time()
        df, banner = S.search_books(q, system, top_k=24)
        sec = time.time() - t
        top = top_text(df) if len(df) else []
        if isinstance(expect, tuple):
            ok = expect[1] in (banner or '').lower()
        else:
            ok = len(df) > 0 and any(re.search(expect, x) for x in top)
        rows.append({'kind': kind, 'query': q, 'ok': ok, 'sec': round(sec, 2), 'banner': (banner or '')[:90],
                     'top': ' || '.join(x.split(' | ')[0][:40] + ' / ' + x.split(' | ')[1][:20] for x in top[:3])})
    res = pd.DataFrame(rows)
    res.to_csv('tests/typos.csv', index=False)
    print(f"{res['ok'].sum()}/{len(res)} misspelled searches found the right books")
    print(f"slowest: {res['sec'].max()} s ({res.loc[res['sec'].idxmax(), 'query']!r})\n")
    for kind, r in res.groupby('kind', sort=False)['ok'].agg(['sum', 'count']).iterrows():
        print(f"  {'OK  ' if r['sum'] == r['count'] else 'FAIL'} {kind:24} {r['sum']}/{r['count']}")
    print('\n=== need fixing ===')
    for _, r in res[~res['ok']].iterrows():
        print(f"  [{r['kind']}] {r['query']!r}\n      message: {r['banner'] or '-'}\n      top: {r['top'] or '(nothing)'}")
