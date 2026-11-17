"""Every kind of thing people type into the search box, with what a good answer looks like.
Run from D:/documents/Smart_Library_V2:   python tests/scenarios.py
A search passes when one of the TOP 3 books matches the expected pattern (title or author),
or, for "say" checks, when the message on the page contains the expected words.
Writes tests/scenarios.csv with every result."""
import re
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import pandas as pd
import search as S
S.live_lookup = lambda *a, **k: []                  # our own catalog only
S.get_better_cover = lambda isbn, title, author, fallback: fallback

# (kind of search, what is typed, expected pattern in a top-3 title/author  OR  ('say', words in the message))
SCENARIOS = [
    # 1-5: titles
    ('Exact title', 'the great gatsby', r'gatsby'),
    ('Exact title', 'to kill a mockingbird', r'mockingbird'),
    ('Exact title', 'pride and prejudice', r'pride and prejudice'),
    ('Part of a title', 'gatsby', r'gatsby'),
    ('Part of a title', 'mockingbird', r'mockingbird'),
    ('Part of a title', 'hunger games', r'hunger games'),
    ('Misspelled title', 'harry poter', r'harry potter'),
    ('Misspelled title', 'moby dik', r'moby'),
    ('Misspelled title', 'pride and prejudise', r'pride and prejudice'),
    ('Misspelled title', 'the hobit', r'hobbit'),
    ('Misspelled title', 'to kil a mokingbird', r'mockingbird'),
    ('Title + extra words', 'harry potter book', r'harry potter'),
    ('Title + extra words', 'the hobbit novel', r'hobbit'),
    ('Unfinished title', 'harry pot', r'harry potter'),
    ('Unfinished title', 'lord of the ri', r'lord of the rings'),
    ('Unfinished title', 'hunger gam', r'hunger games'),
    # 6-12: authors
    ('Full author name', 'stephen king', r'stephen king'),
    ('Full author name', 'agatha christie', r'agatha christie'),
    ('Full author name', 'jane austen', r'jane austen'),
    ('Surname only', 'rowling', r'rowling'),
    ('Surname only', 'tolkien', r'tolkien'),
    ('Surname only', 'christie', r'agatha christie'),
    ('Surname only', 'austen', r'jane austen'),
    ('First name only', 'stephen', r'stephen king'),
    ('First name only', 'stephan', r'stephen king'),
    ('First name only', 'agatha', r'agatha christie'),
    ('First name only', 'jane', r'jane austen|jane eyre'),
    ('First name only', 'dan', r'dan brown'),
    ('Misspelled author', 'roling', r'rowling'),
    ('Misspelled author', 'tolkein', r'tolkien'),
    ('Misspelled author', 'agata christie', r'agatha christie'),
    ('Misspelled author', 'stephen kng', r'stephen king'),
    ('Misspelled author', 'shakspeare', r'shakespeare'),
    ('Initials', 'jk', r'rowling'),
    ('Initials', 'jrr tolkien', r'tolkien'),
    ('Initials', 'cs lewis', r'lewis'),
    ('Initials', 'j.k. rowling', r'rowling'),
    ('Author + title mixed', 'rowling harry potter', r'harry potter'),
    ('Author + title mixed', 'king the shining', r'shining'),
    ('Author + title mixed', 'tolkien hobbit', r'hobbit'),
    ('"books by" phrasing', 'books by dickens', r'dickens'),
    ('"books by" phrasing', 'novels by jane austen', r'austen'),
    ('"books by" phrasing', 'written by agatha christie', r'christie'),
    ('Name swapped', 'king stephen', r'stephen king'),
    ('Name swapped', 'potter harry', r'harry potter'),
    # 13-17: topics, descriptions, study
    ('Genre / topic', 'horror', r'.'),
    ('Genre / topic', 'romance novels', r'.'),
    ('Genre / topic', 'science fiction', r'.'),
    ('Genre / topic', 'history of rome', r'rome|roman'),
    ('Description / goal', 'a sad story about growing up', r'.'),
    ('Description / goal', 'a book about loneliness', r'lonel|alone'),
    ('Description / goal', 'easy fantasy book about friendship and adventure', r'.'),
    ('Study subject', 'calculus', r'calculus'),
    ('Study subject', 'organic chemistry', r'organic chemistry'),
    ('Study subject', 'python programming', r'python'),
    ('Study subject', 'machine learning for beginners', r'machine learning'),
    ('Study subject', 'data structures and algorithms', r'data structures|algorithm'),
    ('Study short form', 'ai', ('say', 'artificial intelligence')),
    ('Study short form', 'dsa', ('say', 'data structures')),
    ('Study short form', 'dbms', ('say', 'database')),
    ('Level words', 'beginner calculus', r'calculus'),
    ('Level words', 'introduction to physics', r'physic'),
    # 18-22: series, characters, numbers, ISBN, other languages
    ('Series name', 'lord of the rings', r'lord of the rings|fellowship|two towers|return of the king'),
    ('Series name', 'narnia', r'narnia|lion, the witch'),
    ('Series name', 'game of thrones', r'game of thrones|song of ice'),
    ('Character name', 'sherlock holmes', r'sherlock|holmes'),
    ('Character name', 'hermione', r'harry potter|hermione'),
    ('Character name', 'frodo', r'lord of the rings|fellowship|hobbit|frodo'),
    ('Number / year title', '1984', r'1984|orwell'),
    ('Number / year title', 'catch 22', r'catch-?22'),
    ('Number / year title', 'fahrenheit 451', r'fahrenheit'),
    ('ISBN', '0439139597', r'harry potter'),
    ('ISBN', '9780439139595', r'harry potter'),
    ('Accents / other language', 'garcía márquez', r'garc[ií]a m[aá]rquez'),
    ('Accents / other language', 'les miserables', r'mis[eé]rables'),
    # 23-27: typing quirks and edge cases
    ('CAPITALS / punctuation', 'HARRY POTTER', r'harry potter'),
    ('CAPITALS / punctuation', 'harry-potter', r'harry potter'),
    ('Missing spaces', 'harrypotter', r'harry potter'),
    ('Missing spaces', 'lordoftherings', r'lord of the rings'),
    ('Short real title', 'it', r'stephen king|^it\b'),
    ('Ambiguous word', 'king', r'stephen king|king'),
    ('Ambiguous word', 'brown', r'dan brown|brown'),
    ('Plural / word form', 'vampires', r'vampire'),
    ('Plural / word form', 'dragon', r'dragon'),
    # two meanings at once (an author AND a title word / typo): the more popular one first, both shown
    ('Two meanings', 'harr', r'harry potter'),
    ('Two meanings', 'sherlock', r'sherlock holmes|adventures of sherlock|hound of the baskervilles|sherlock'),
    ('Two meanings', 'potter', r'harry potter'),
    ('Two meanings', 'herry potter', r'harry potter'),
    ('Two meanings', 'kathleen rowling', r'rowling'),
    ('Two meanings', 'arthur', r'arthur c\.? clarke|conan doyle|arthur miller'),
    ('Two meanings', 'ronald tolkien', r'^(the )?(fellowship|hobbit|two towers|return of the king|lord of the rings)'),
    ('Not in our library', 'project hail mary', ('say', "isn't in our library")),
    ('Nonsense', 'asdfghjk', ('say', 'no books match')),
    ('Nonsense', 'qwertyuiop zxcv', ('say', 'no books match')),
]


def top_text(df, n=3):
    rows = df.head(n).fillna('')
    return [S.fold(f"{r.get('title', '')} | {r.get('author', '')}") for _, r in rows.iterrows()]


if __name__ == '__main__':
    system = S.load_search_system('data/processed')
    rows = []
    for kind, q, expect in SCENARIOS:
        df, banner = S.search_books(q, system, top_k=24)
        banner = banner or ''
        top = top_text(df) if len(df) else []
        if isinstance(expect, tuple):
            ok = expect[1].lower() in banner.lower()
        else:
            ok = len(df) > 0 and any(re.search(expect, t) for t in top)
        rows.append({'kind': kind, 'query': q, 'ok': ok, 'n': len(df), 'banner': banner[:90],
                     'top3': ' || '.join(t[:45] for t in top)})
    res = pd.DataFrame(rows)
    res.to_csv('tests/scenarios.csv', index=False)
    print(f"{res['ok'].sum()}/{len(res)} searches gave a good answer\n")
    by = res.groupby('kind', sort=False)['ok'].agg(['sum', 'count'])
    for kind, r in by.iterrows():
        print(f"  {'OK  ' if r['sum'] == r['count'] else 'FAIL'} {kind:28} {r['sum']}/{r['count']}")
    print('\n=== searches that need fixing ===')
    for _, r in res[~res['ok']].iterrows():
        print(f"  [{r['kind']}] {r['query']!r}\n      message: {r['banner'] or '-'}\n      top 3:   {r['top3'] or '(nothing)'}")
