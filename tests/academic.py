"""Academic / textbook searches: textbook authors (surname, full name, initials, typos), textbook titles
(exact, partial, misspelled), free OpenStax books, subjects + level words, and author + subject together.
Run from D:/documents/Smart_Library_V2:   python tests/academic.py
A search passes when one of the TOP 5 books matches the expected pattern (title or author)."""
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
    # textbook authors: as typed by students
    ('Author', 'cormen', r'cormen|introduction to algorithms'),
    ('Author', 'thomas cormen', r'cormen'),
    ('Author', 'knuth', r'knuth'),
    ('Author', 'donald knuth', r'knuth'),
    ('Author', 'sedgewick', r'sedgewick'),
    ('Author', 'silberschatz', r'silberschatz'),
    ('Author', 'kernighan', r'kernighan'),
    ('Author', 'kernighan ritchie', r'kernighan|ritchie|c programming'),
    ('Author', 'sipser', r'sipser'),
    ('Author', 'kreyszig', r'kreyszig'),
    ('Author', 'apostol', r'apostol'),
    ('Author', 'axler', r'axler|linear algebra'),
    ('Author', 'sheldon axler', r'axler'),
    ('Author', 'mankiw', r'mankiw'),
    ('Author', 'keynes', r'keynes'),
    ('Author', 'john maynard keynes', r'keynes'),
    ('Author', 'stallings', r'stallings'),
    ('Author', 'william stallings', r'stallings'),
    ('Author', 'tanenbaum computer networks', r'tanenbaum'),
    ('Author', 'serway physics', r'serway'),
    ('Author', 'giancoli', r'giancoli'),
    ('Author', 'zill differential equations', r'zill|differential'),
    ('Author', 'hawking', r'hawking'),
    ('Author', 'stephen hawking', r'hawking'),
    ('Author', 'carl sagan', r'sagan'),
    ('Author', 'richard dawkins', r'dawkins'),
    ('Author', 'richard feynman', r'feynman'),
    ('Author', 'feynman', r'feynman'),
    ('Author', 'kahneman', r'kahneman'),
    ('Author', 'hofstadter', r'hofstadter'),
    ('Author', 'allen downey', r'downey'),
    ('Author', 'jurafsky', r'jurafsky'),
    ('Author', 'openstax', r'openstax'),
    ('Author', 'philip kotler', r'kotler'),
    ('Author', 'brian greene', r'greene'),
    # initials
    ('Author initials', 'd e knuth', r'knuth'),
    ('Author initials', 'b w kernighan', r'kernighan'),
    ('Author initials', 'th cormen', r'cormen'),
    # author typos
    ('Author typo', 'cormon', r'cormen'),
    ('Author typo', 'thomas cormin', r'cormen'),
    ('Author typo', 'kunth', r'knuth'),
    ('Author typo', 'donald kunth', r'knuth'),
    ('Author typo', 'sedgwick', r'sedgewick'),
    ('Author typo', 'silberschats', r'silberschatz'),
    ('Author typo', 'silbershatz', r'silberschatz'),
    ('Author typo', 'kernigan', r'kernighan'),
    ('Author typo', 'sipsr', r'sipser'),
    ('Author typo', 'kreyzig', r'kreyszig'),
    ('Author typo', 'kreysig', r'kreyszig'),
    ('Author typo', 'mankew', r'mankiw'),
    ('Author typo', 'keyns', r'keynes'),
    ('Author typo', 'stalings', r'stallings'),
    ('Author typo', 'tanenbaun', r'tanenbaum'),
    ('Author typo', 'hawkings', r'hawking'),
    ('Author typo', 'stephen hawkin', r'hawking'),
    ('Author typo', 'carl sagen', r'sagan'),
    ('Author typo', 'richard dawkin', r'dawkins'),
    ('Author typo', 'feynmann', r'feynman'),
    ('Author typo', 'richard fenyman', r'feynman'),
    ('Author typo', 'kahnemann', r'kahneman'),
    ('Author typo', 'daniel kahnemen', r'kahneman'),
    ('Author typo', 'hofstader', r'hofstadter'),
    ('Author typo', 'open stax', r'openstax'),
    ('Author typo', 'openstacks', r'openstax'),
    # textbook titles
    ('Title', 'introduction to algorithms', r'introduction to algorithms'),
    ('Title', 'operating system concepts', r'operating system concepts'),
    ('Title', 'the c programming language', r'c programming language'),
    ('Title', 'linear algebra done right', r'linear algebra done right'),
    ('Title', 'structure and interpretation of computer programs', r'structure and interpretation'),
    ('Title', 'introduction to the theory of computation', r'theory of computation'),
    ('Title', 'advanced engineering mathematics', r'advanced engineering mathematics'),
    ('Title', 'the art of computer programming', r'art of computer programming'),
    ('Title', 'design patterns', r'design patterns'),
    ('Title', 'a brief history of time', r'brief history of time'),
    ('Title', 'the selfish gene', r'selfish gene'),
    ('Title', 'thinking fast and slow', r'thinking,? fast and slow'),
    ('Title', 'godel escher bach', r'godel,? escher'),
    ('Title', 'the general theory of employment interest and money', r'general theory'),
    ('Title', 'calculus volume 1', r'calculus volume 1|calculus, volume 1'),
    ('Title', 'biology 2e', r'biology 2e'),
    ('Title', 'principles of economics', r'principles of economics'),
    ('Title', 'think python', r'think python'),
    ('Title', 'speech and language processing', r'speech and language processing'),
    ('Title', 'eloquent javascript', r'eloquent javascript'),
    ('Title', 'freakonomics', r'freakonomics'),
    ('Title', 'the elegant universe', r'elegant universe'),
    # textbook titles with typos
    ('Title typo', 'introduction to algoritms', r'introduction to algorithms'),
    ('Title typo', 'intro to algorithms', r'introduction to algorithms|algorithm'),
    ('Title typo', 'operating sytem concepts', r'operating system concepts'),
    ('Title typo', 'the c programing language', r'c programming language'),
    ('Title typo', 'linear algebra done rite', r'linear algebra done right'),
    ('Title typo', 'linaer algebra', r'linear algebra'),
    ('Title typo', 'theory of computaton', r'theory of computation'),
    ('Title typo', 'advanced engineering mathematic', r'advanced engineering mathematics'),
    ('Title typo', 'advance engineering maths', r'advanced engineering mathematics|engineering mathematics'),
    ('Title typo', 'a breif history of time', r'brief history of time'),
    ('Title typo', 'the selfsh gene', r'selfish gene'),
    ('Title typo', 'thinking fast and slwo', r'thinking,? fast and slow'),
    ('Title typo', 'calculas volume 1', r'calculus'),
    ('Title typo', 'biolgy 2e', r'biology'),
    ('Title typo', 'principals of economics', r'principles of economics'),
    ('Title typo', 'think pyhton', r'think python'),
    ('Title typo', 'eloquent javascipt', r'eloquent javascript'),
    ('Title typo', 'freakanomics', r'freakonomics'),
    ('Title typo', 'the elegent universe', r'elegant universe'),
    ('Title typo', 'organic chemsitry', r'organic chemistry'),
    ('Title typo', 'univercity physics', r'university physics|physics'),
    ('Title typo', 'colege algebra', r'college algebra'),
    # subjects with level words / course codes
    ('Subject', 'calculus for beginners', r'calculus'),
    ('Subject', 'intro to statistics', r'statistic'),
    ('Subject', 'introductory physics', r'physics'),
    ('Subject', 'advanced calculus', r'calculus'),
    ('Subject', 'discrete math', r'discrete'),
    ('Subject', 'differential equations', r'differential equations'),
    ('Subject', 'diffrential equations', r'differential equations'),
    ('Subject', 'thermodynamics', r'thermodynamic'),
    ('Subject', 'thermodynamcis', r'thermodynamic'),
    ('Subject', 'quantum mechanics', r'quantum'),
    ('Subject', 'quantum mechanic', r'quantum'),
    ('Subject', 'microeconomics', r'microeconomic'),
    ('Subject', 'micro economics', r'microeconomic|economics'),
    ('Subject', 'macroeconomics textbook', r'macroeconomic'),
    ('Subject', 'computer architecture', r'computer architecture|computer organization'),
    ('Subject', 'computer archtecture', r'computer architecture|computer organization'),
    ('Subject', 'object oriented programming', r'object'),
    ('Subject', 'java programming', r'java'),
    ('Subject', 'javascript for beginners', r'javascript'),
    ('Subject', 'c++ programming', r'c\+\+'),
    ('Subject', 'anatomy and physiology', r'anatomy'),
    ('Subject', 'anatomy and phisiology', r'anatomy'),
    ('Subject', 'financial accounting', r'accounting'),
    ('Subject', 'acounting principles', r'accounting'),
    ('Subject', 'marketing management', r'marketing'),
    ('Subject', 'psychology 2e', r'psychology'),
    ('Subject', 'sociology textbook', r'sociology'),
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
        if kind == 'Author typo':
            # a typo close to two names: the more-read one comes first, the other must be ON THE PAGE
            # as its own "Did you mean" / "similar name" section
            top = top_text(df, n=len(df)) if len(df) else []
        ok = len(df) > 0 and any(re.search(expect, x) for x in top)
        rows.append({'kind': kind, 'query': q, 'ok': ok, 'sec': round(sec, 2), 'banner': (banner or '')[:90],
                     'top': ' || '.join(x[:60] for x in top[:3])})
    res = pd.DataFrame(rows)
    res.to_csv('tests/academic.csv', index=False)
    print(f"{res['ok'].sum()}/{len(res)} academic searches found the right books")
    print(f"slowest: {res['sec'].max()} s ({res.loc[res['sec'].idxmax(), 'query']!r})\n")
    for kind, r in res.groupby('kind', sort=False)['ok'].agg(['sum', 'count']).iterrows():
        print(f"  {'OK  ' if r['sum'] == r['count'] else 'FAIL'} {kind:16} {r['sum']}/{r['count']}")
    print('\n=== need fixing ===')
    for _, r in res[~res['ok']].iterrows():
        print(f"  [{r['kind']}] {r['query']!r}\n      message: {r['banner'] or '-'}\n      top: {r['top'] or '(nothing)'}")
