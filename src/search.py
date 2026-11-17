import pandas as pd
import numpy as np
import os
import re
import unicodedata
import bisect
import math
import heapq
import faiss
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process, utils
from sentence_transformers import SentenceTransformer
import requests
import html
from spellchecker import SpellChecker
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


spell = SpellChecker()

ENABLE_LIVE_SUPPLEMENT = True          # set False to turn off the "Also found online" section
LIVE_LABEL = 'Also found online (Open Library)'
live_executor = ThreadPoolExecutor(max_workers=2)

# Open Library asks apps to identify themselves. Set SMART_LIBRARY_CONTACT (e.g. an email) to add a contact.
_contact = os.environ.get('SMART_LIBRARY_CONTACT', '').strip()
USER_AGENT = 'SmartLibraryStudentProject/1.0' + (f' (contact: {_contact})' if _contact else '')


# ---------- text helpers ----------

def fold(s):
    """Lowercase and strip accents: 'García Márquez' -> 'garcia marquez'."""
    s = str(s)
    if s.isascii():                       # most titles: nothing to strip, skip the slow unicode work
        return s.lower()
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def squash_initials(s):
    """'j k rowling' -> 'jk rowling' so spaced and unspaced initials compare equal."""
    out, buf = [], ''
    for tok in s.split():
        if len(tok) == 1:
            buf += tok
        else:
            if buf:
                out.append(buf)
                buf = ''
            out.append(tok)
    if buf:
        out.append(buf)
    return ' '.join(out)


def proc(s):
    """Normal form used for fuzzy matching of titles and authors."""
    return squash_initials(utils.default_process(fold(s)))


def correct_query(query, vocab=None):
    """Spell-correct, but leave alone any word that really occurs in our catalog (e.g. 'moby')."""
    def fix(w):
        if len(w) <= 2 or (vocab and fold(w) in vocab):
            return w
        c = spell.correction(w)
        # accept only a word our books actually contain, and only if it is close to what was typed:
        # "calculas" -> calculus yes; "frodo" -> "fordo" no (no book has it); "zxcv" -> "zach" no (not close)
        if c and (not vocab or fold(c) in vocab) and fuzz.ratio(fold(w), fold(c)) >= 80:
            return c
        return w
    return ' '.join(fix(w) for w in query.split())


CATEGORY_GROUPS = {
    'Fiction': [r'(?<!non)(?<!non-)(?<!non )fiction', 'novel', 'stories', 'literature'],   # not "Nonfiction"
    "Children's": ['juvenile fiction', 'children'],
    'Computer Science': ['computer', 'programming', 'software'],
    'Mathematics': ['mathematics', 'math', 'calculus'],
    'Engineering': ['engineering', 'mechanical', 'electrical'],
    'Science': [r'science(?![\s-]*fiction)', 'physics', 'biology', 'chemistry'],   # not "Science-Fiction"
}

STOPWORDS = {'a', 'an', 'the', 'give', 'me', 'for', 'something', 'like', 'please', 'show', 'find', 'i', 'want', 'to', 'read', 'book', 'books', 'about', 'story', 'stories'}


RELATED_COS = 0.50       # meaning-similarity needed to count as "related" (gibberish never passes 0.48)


def filter_label(category_filter, free_only):
    """'free Computer Science books', 'Fiction books', 'free books' ..."""
    cat = category_filter if category_filter in CATEGORY_GROUPS else ''
    return ' '.join(x for x in ('free' if free_only else '', cat, 'books') if x)


def strip_stopwords(query):
    words = [w for w in query.lower().split() if w not in STOPWORDS]
    return ' '.join(words) if words else query


def make_dedup_key(title, author):
    t = re.sub(r'\(.*?\)', ' ', str(title).lower())
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'^(the|a|an)\s+', '', t.strip())
    t = ' '.join(t.split())
    a = re.sub(r'[^\w\s]', ' ', str(author).lower())
    a = ' '.join(sorted(a.split()))
    return t + '|' + a


def to_isbn13(isbn):
    isbn = str(isbn).strip().replace('-', '')
    if len(isbn) == 13:
        return isbn
    if len(isbn) != 10:
        return None
    core = '978' + isbn[:9]
    if not core.isdigit():
        return None
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
    return core + str((10 - total % 10) % 10)


# ---------- loading ----------

def load_search_system(data_dir):
    books = pd.read_parquet(f'{data_dir}/books_with_mood.parquet')
    work_dedup = pd.read_parquet(f'{data_dir}/books_search_ready.parquet')
    embeddings = np.load(f'{data_dir}/book_embedding.npy')

    books['title'] = books['title'].apply(lambda x: html.unescape(x) if pd.notna(x) else x)
    books['author'] = books['author'].apply(lambda x: html.unescape(x) if pd.notna(x) else x)
    work_dedup['title'] = work_dedup['title'].apply(lambda x: html.unescape(x) if pd.notna(x) else x)
    work_dedup['author'] = work_dedup['author'].apply(lambda x: html.unescape(x) if pd.notna(x) else x)

    books['author'] = books['author'].fillna('Unknown author')
    work_dedup['author'] = work_dedup['author'].fillna('Unknown author')
    books['categories'] = books['categories'].fillna('')
    work_dedup['categories'] = work_dedup['categories'].fillna('')

    books['cover_url'] = books['cover_url'].fillna('')
    work_dedup['cover_url'] = work_dedup['cover_url'].fillna('')

    work_dedup['description'] = (
        work_dedup['description'].str.replace(r'\[\d+\]', '', regex=True).fillna('')
    )

    # accent-free copies used for keyword matching
    work_dedup['title_f'] = work_dedup['title'].fillna('').map(fold)
    work_dedup['author_f'] = work_dedup['author'].fillna('').map(fold)

    work_dedup['search_text'] = (
        work_dedup['title_f'] + ' ' +
        work_dedup['author_f'] + ' ' +
        work_dedup['categories'].fillna('').map(fold)
    )

    work_dedup['dedup_key'] = [make_dedup_key(t, a) for t, a in zip(work_dedup['title'], work_dedup['author'])]


    # extra free books (OpenStax, CSV imports...), made by scripts/import_books.py
    extra_read, extra_vecs, ex_new = {}, None, None
    extra_path = f'{data_dir}/extra_books.parquet'
    if os.path.exists(extra_path) and os.path.exists(f'{data_dir}/extra_embeddings.npy'):
        ex = pd.read_parquet(extra_path)
        ex_vec = np.load(f'{data_dir}/extra_embeddings.npy')
        assert len(ex) == len(ex_vec), "extra_books and extra_embeddings are out of sync: run scripts/import_books.py again"
        ex['dedup_key'] = [make_dedup_key(t, a) for t, a in zip(ex['title'], ex['author'])]
        extra_read = {k: u for k, u in zip(ex['dedup_key'], ex['read_url']) if u}
        is_new = ~ex['dedup_key'].isin(set(work_dedup['dedup_key']))
        ex_new, extra_vecs = ex[is_new.values].copy(), ex_vec[is_new.values]
        ex_new['title_f'] = ex_new['title'].map(fold)
        ex_new['author_f'] = ex_new['author'].map(fold)
        ex_new['search_text'] = ex_new['title_f'] + ' ' + ex_new['author_f'] + ' ' + ex_new['categories'].map(fold)
        work_dedup = pd.concat([work_dedup, ex_new.drop(columns=['read_url'])], ignore_index=True)
        print(f"Extra free books added: {len(ex_new)} (links added to {len(extra_read) - len(ex_new)} existing books)")

    # free full-text links (Project Gutenberg), made by scripts/add_gutenberg_links.py
    links_path = f'{data_dir}/gutenberg_links.parquet'
    if os.path.exists(links_path):
        gl = pd.read_parquet(links_path).drop_duplicates('dedup_key').set_index('dedup_key')['read_url']
        work_dedup['read_url'] = work_dedup['dedup_key'].map(gl).fillna('')
    else:
        work_dedup['read_url'] = ''

    work_dedup['read_url'] = [extra_read.get(k) or r for k, r in zip(work_dedup['dedup_key'], work_dedup['read_url'])]
    print(f"Books with a free 'Read' link: {(work_dedup['read_url'] != '').sum()}")

    ratings = pd.read_parquet(f'{data_dir}/ratings_clean.parquet', columns=['ISBN', 'Book-Rating'])
    isbn_map = {i: to_isbn13(i) for i in ratings['ISBN'].unique()}
    ratings['isbn13'] = ratings['ISBN'].map(isbn_map)

    # map every edition's ISBN to its work key (books holds all editions)
    books['dedup_key'] = [make_dedup_key(t, a) for t, a in zip(books['title'], books['author'])]
    key_by_isbn = dict(zip(books['isbn13'], books['dedup_key']))
    ratings['dedup_key'] = ratings['isbn13'].map(key_by_isbn)

    explicit = ratings[(ratings['Book-Rating'] > 0) & ratings['dedup_key'].notna()]
    stats = explicit.groupby('dedup_key')['Book-Rating'].agg(avg_rating='mean', n_ratings='count')
    stats['avg_rating'] = stats['avg_rating'] / 2   # convert 1-10 to a 5-star scale

    work_dedup['avg_rating'] = work_dedup['dedup_key'].map(stats['avg_rating'])
    work_dedup['n_ratings'] = work_dedup['dedup_key'].map(stats['n_ratings']).fillna(0).astype(int)
    work_dedup.loc[work_dedup['n_ratings'] < 5, 'avg_rating'] = np.nan

    print(f"Books with a displayable rating: {work_dedup['avg_rating'].notna().sum()}")

    assert len(books) == embeddings.shape[0], "Books and embeddings out of sync"

    isbn13_by_position = books['isbn13'].values
    if extra_vecs is not None and len(extra_vecs):
        embeddings = np.vstack([embeddings, extra_vecs.astype(embeddings.dtype)])
        isbn13_by_position = np.concatenate([isbn13_by_position, ex_new['isbn13'].values])

    # Initials: "J.K. Rowling" / "J. K. Rowling" also get the word "jk", "J. R. R. Tolkien" gets "jrr",
    # so typing "jk" or "jrr tolkien" finds them. Only the new initials word is added (rankings stay the same).
    initials = {}
    for a in work_dedup['author'].fillna('').unique():
        have = set(re.sub(r'[^\w\s]', ' ', fold(a)).split())
        initials[a] = ' '.join(t for t in proc(a).split() if t not in have and len(t) >= 2)
    work_dedup['author_initials'] = work_dedup['author'].fillna('').map(initials)
    work_dedup['search_text'] = (work_dedup['search_text'] + ' ' + work_dedup['author_initials']).str.rstrip()
    # titles + authors in one column: used to tell an unfinished word ("harr") from a finished one ("king")
    work_dedup['name_f'] = work_dedup['title_f'] + ' | ' + work_dedup['author_f'] + ' ' + work_dedup['author_initials']
    # the source data has some broken characters ("�rden del Fénix"): drop the replacement character
    for col in ('title', 'author'):
        work_dedup[col] = work_dedup[col].str.replace('�', '', regex=False)

    tokenized_corpus = [text.split() for text in work_dedup['search_text']]
    bm25 = BM25Okapi(tokenized_corpus)
    word_counts = Counter(w for doc in tokenized_corpus for w in set(doc))
    vocab = {w for w, c in word_counts.items() if c >= 3}   # words seen in at least 3 books

    # Word index: word -> ids of the books whose title / author / category contain it. Built once, it makes
    # "is the last word unfinished?" and "how many of the search words does each book have?" instant
    # instead of re-scanning all books on every search.
    postings = {}
    for i, text in enumerate(work_dedup['search_text']):
        for w in set(re.findall(r'\w+', text)):
            postings.setdefault(w, []).append(i)
    word_index = {w: np.asarray(ids, dtype=np.int32) for w, ids in postings.items()}
    del postings
    index_words = sorted(word_index)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype('float32'))

    model_embed = SentenceTransformer('all-MiniLM-L6-v2')

    books_by_author = work_dedup['author_norm'].value_counts()
    # fold rare (1-2 book) author spellings into a much more common close match.
    # Batched with rapidfuzz.cdist: same result as one extractOne per author, but seconds instead of hours.
    alias_map = {}
    names, vals = books_by_author.index.to_numpy(), books_by_author.to_numpy()
    for count in (1, 2):
        queries, choices = names[vals == count], names[vals >= count * 3]
        if len(queries) == 0 or len(choices) == 0:
            continue
        for start in range(0, len(queries), 2000):
            q = queries[start:start + 2000]
            m = process.cdist(q, choices, scorer=fuzz.ratio, score_cutoff=90, workers=-1, dtype=np.uint8)
            best = m.argmax(axis=1)              # first best = most common author, like extractOne
            for a, b, s in zip(q, best, m[np.arange(len(q)), best]):
                if s >= 90:
                    alias_map[a] = choices[b]
    if alias_map:
        canonical_display = (work_dedup.sort_values('rating_count', ascending=False)
                              .drop_duplicates('author_norm').set_index('author_norm')['author'])
        # one dict lookup per row (Series.replace with a big dict does one full pass per key: 78s -> <1s)
        work_dedup['author_norm'] = [alias_map.get(a, a) for a in work_dedup['author_norm']]
        work_dedup['author'] = work_dedup['author_norm'].map(canonical_display).fillna(work_dedup['author'])

    author_list = [a for a in work_dedup['author'].dropna().unique().tolist()
                   if len(a) >= 4 and a != 'Unknown author']
    title_list = [t for t in work_dedup['title'].dropna().unique().tolist() if len(t) >= 4]
    author_proc = [proc(a) for a in author_list]
    title_proc = [proc(t) for t in title_list]

    work_dedup['surname'] = [fold(a.split()[-1]) if a.split() else '' for a in work_dedup['author_norm'].fillna('')]
    surname_list = [s for s in work_dedup['surname'].unique() if len(s) >= 4 and s != 'author']

    # "How well-known" = number of books + number of reader ratings. One shared measure, used whenever the
    # search has to choose between several authors ("twain" -> Mark Twain, "shakspeare" -> Shakespeare).
    by_author = work_dedup.groupby('author_norm')
    author_fame = (by_author.size() + by_author['rating_count'].sum()).to_dict()
    by_surname = work_dedup.groupby('surname')
    surname_fame = (by_surname.size() + by_surname['rating_count'].sum()).to_dict()
    ranked_authors = (work_dedup.assign(_fame=work_dedup['author_norm'].map(author_fame))
                      .sort_values(['_fame', 'rating_count'], ascending=False))
    surname_best_author = ranked_authors.drop_duplicates('surname').set_index('surname')['author_norm'].to_dict()
    # first names too: "stephen" -> Stephen King, Stephen Hawking ...; "agatha" -> Agatha Christie
    ranked_authors['firstname'] = [fold(str(a).split()[0]) if str(a).split() else '' for a in ranked_authors['author_norm']]
    one_per_author = ranked_authors.drop_duplicates('author_norm')
    authors_by_name = {}                                        # name word -> best-known authors with it
    for col in ('firstname', 'surname'):
        for name, group in one_per_author.groupby(col, sort=False):
            if len(name) >= 3 and name.isalpha() and name not in ('unknown', 'author', 'anonymous'):
                authors_by_name.setdefault(name, []).extend(group['author_norm'].head(10).tolist())
    authors_by_name = {k: sorted(dict.fromkeys(v), key=lambda a: -author_fame.get(a, 0))[:10]
                       for k, v in authors_by_name.items()}
    name_words = sorted(authors_by_name)
    author_display = ranked_authors.drop_duplicates('author_norm').set_index('author_norm')['author'].to_dict()
    author_norm_of = work_dedup.drop_duplicates('author').set_index('author')['author_norm'].to_dict()

    return {
        'books': books, 'work_dedup': work_dedup, 'embeddings': embeddings,
        'bm25': bm25, 'faiss_index': faiss_index, 'model_embed': model_embed,
        'isbn13_by_position': isbn13_by_position,
        'author_list': author_list, 'author_proc': author_proc,
        'title_list': title_list, 'title_proc': title_proc,
        'surname_list': surname_list,
        'dedup_keys': set(work_dedup['dedup_key']),
        'vocab' : vocab,
        'word_df': word_counts,                 # word -> number of books containing it
        'word_index': word_index, 'index_words': index_words,
        'author_fame': author_fame, 'surname_fame': surname_fame,
        'surname_best_author': surname_best_author, 'author_display': author_display,
        'authors_by_name': authors_by_name, 'name_words': name_words,
        'author_norm_of': author_norm_of,
    }


# ---------- helpers ----------

def suggest_correction(query, choices_df, limit=10, threshold=65):
    title_matches = process.extract(query, choices_df['title'].tolist(), scorer=fuzz.WRatio, processor=utils.default_process, limit=limit)
    author_matches = process.extract(query, choices_df['author'].dropna().unique().tolist(), scorer=fuzz.WRatio, processor=utils.default_process, limit=limit)
    scored = []
    for text, fuzzy_score, idx in title_matches:
        if fuzzy_score < threshold:
            continue
        pop = choices_df[choices_df['title'] == text]['rating_count'].max()
        scored.append((text, fuzzy_score + min(pop, 200), 'title'))
    for text, fuzzy_score, idx in author_matches:
        if fuzzy_score < threshold:
            continue
        pop = choices_df[choices_df['author'] == text]['rating_count'].max()
        scored.append((text, fuzzy_score + min(pop, 200), 'author'))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(text, field) for text, score, field in scored[:3]]


def live_lookup(query, limit=5, timeout=10):
    url = "https://openlibrary.org/search.json"
    params = {'q': query, 'limit': limit}
    headers = {'User-Agent': USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return []
        data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return []

    results = []
    for doc in data.get('docs', []):
        isbn_list = doc.get('isbn', [])
        results.append({
            'title': doc.get('title'),
            'author': ', '.join(doc.get('author_name', [])) if doc.get('author_name') else None,
            'isbn13': isbn_list[0] if isbn_list else None,
            'ol_key': doc.get('key'),
            'cover_url': f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-L.jpg" if doc.get('cover_i') else None,
            'source': 'live_lookup'
        })
    return results


# ---------- understanding what the user typed ----------
#
# Every decision about HOW to read a query lives in understand_query(), in one fixed order:
#   1. unfinished last word  ("harr", "harry pot", "tolk")    -> match it as "starts with"
#   2. a surname, typed right or with a typo ("rowling", "roling", "shakspeare")
#   3. a full author name ("stephen king", "jk roling", "j k rolling")
#   4. a misspelled title ("harry poter", "moby dik")          -> search that title
#   5. otherwise: the words as typed (unknown words spell-checked)
# When it must choose between several authors it always uses the same measure: how well-known
# they are (books + reader ratings), from the tables built at startup.

EMPTY_IDS = np.zeros(0, dtype=np.int32)
DESCRIPTION_WORDS = {'a', 'the', 'book', 'about', 'story', 'something', 'like', 'that'}


def docs_with(sys, word):
    """Ids of the books whose title / author / category contain this exact word."""
    return sys['word_index'].get(word, EMPTY_IDS)


def completions(sys, prefix, limit=200):
    """Longer words that start with `prefix` ("harr" -> harry, harris, ...), most common first."""
    words = sys['index_words']
    lo, hi = bisect.bisect_left(words, prefix), bisect.bisect_left(words, prefix + '￿')
    found = [w for w in words[lo:hi] if w != prefix]
    if len(found) > limit:
        found = heapq.nlargest(limit, found, key=lambda w: len(sys['word_index'][w]))
    return found


def docs_starting_with(sys, prefix):
    """Ids of the books that have a word starting with `prefix` (the word itself included)."""
    parts = [docs_with(sys, prefix)] + [docs_with(sys, w) for w in completions(sys, prefix)]
    return np.unique(np.concatenate(parts)) if parts else EMPTY_IDS


def is_unfinished(sys, words_typed):
    """True if the last word looks half-typed: far more books continue it than contain it whole."""
    if not 1 <= len(words_typed) <= 4:
        return False
    last = words_typed[-1]
    if len(last) < 2 or not last.isalpha() or last in STOPWORDS:
        return False
    base = None                                     # books that have all the earlier words
    for w in (p for t in words_typed[:-1] for p in re.findall(r'\w+', t)):
        ids = docs_with(sys, w)
        base = ids if base is None else np.intersect1d(base, ids, assume_unique=True)
        if len(base) == 0:
            return False
    within = (lambda ids: len(ids)) if base is None else (lambda ids: len(np.intersect1d(base, ids, assume_unique=True)))
    longer_words = completions(sys, last)
    if not longer_words:
        return False
    whole = within(docs_with(sys, last))
    longer = within(np.unique(np.concatenate([docs_with(sys, w) for w in longer_words])))
    if base is not None and whole == 0 and longer >= 1:
        return True                                  # "hunger gam": no "gam" with "hunger", but "games" is
    return whole < 50 and longer >= 3 and longer >= 3 * whole


def split_joined_words(sys, word):
    """'lordoftherings' -> ['lord', 'of', 'the', 'rings'] using words our books contain (fewest, most common)."""
    wdf, n = sys['word_df'], len(word)
    best = [(0.0, [])] + [(float('inf'), None)] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            piece, prev = word[j:i], best[j]
            if prev[1] is None:
                continue
            count = wdf.get(piece, 0)
            if count < 5 or (len(piece) == 1 and piece not in ('a', 'i')):
                continue
            cost = prev[0] + 10 - math.log(count)        # each extra word costs more: prefer few, common words
            if cost < best[i][0]:
                best[i] = (cost, prev[1] + [piece])
    words = best[n][1]
    return words if words and len(words) >= 2 else None


def same_person(a, b):
    """'j r r tolkien' and 'jrr tolkien' / 'john ronald reuel tolkien': same surname, same first initial."""
    pa, pb = proc(a).split(), proc(b).split()
    return bool(pa and pb) and pa[-1] == pb[-1] and pa[0][0] == pb[0][0]


def other_authors(sys, names, main, limit=5):
    """Well-known authors sharing a name with `main`, each real person once (no spelling duplicates)."""
    chosen = []
    for a in names:
        if sys['author_fame'].get(a, 0) >= 20 and not same_person(a, main) and not any(same_person(a, c) for c in chosen):
            chosen.append(a)
        if len(chosen) == limit:
            break
    return chosen


def pick_name(sys, word, allow_typo):
    """One typed name -> the name it most likely means, looking at first names AND surnames.
    Exact names are trusted unless a near spelling belongs to a far better-known author
    ("stephan" -> stephen = Stephen King, "shakspeare" -> shakespeare). Typos are fixed only when allowed.
    Returns {'name', 'score', 'exact'} or None."""
    abn, fame = sys['authors_by_name'], sys['author_fame']
    fame_of = lambda name: fame.get(abn[name][0], 0) if name in abn else 0
    near = [(nm, sc) for nm, sc, _ in process.extract(word, sys['name_words'], scorer=fuzz.ratio,
                                                      score_cutoff=85, limit=8) if nm != word]
    if word in abn:
        better = [(nm, sc) for nm, sc in near if fame_of(nm) >= 50 and fame_of(nm) >= 10 * max(fame_of(word), 1)]
        if better:
            nm, sc = max(better, key=lambda x: fame_of(x[0]))
            return {'name': nm, 'score': sc, 'exact': False}
        return {'name': word, 'score': 100, 'exact': True}
    if allow_typo and near:
        top = max(sc for _, sc in near)
        nm, sc = max([x for x in near if x[1] >= top - 8], key=lambda x: fame_of(x[0]))   # near-tied spellings
        return {'name': nm, 'score': sc, 'exact': False}
    return None


def titles_meaning(sys, original_query, reading):
    """Books whose TITLE contains the typed words: the other meaning of a name-like query.
    The last word may be unfinished ("harr" -> Harry...), and in a multi-word query one word may be
    misspelled ("herry potter" -> Harry Potter)."""
    wd = sys['work_dedup']
    words = [w for w in re.findall(r'\w+', fold(original_query)) if w not in STOPWORDS and len(w) >= 2]
    if not words:
        return wd.iloc[0:0]
    prefix = reading.get('title_prefix')

    def ids_for(w):
        return docs_starting_with(sys, w) if w == prefix else docs_with(sys, w)

    def match(ws, typo_at=None):
        cand = None
        for i, w in enumerate(ws):
            if i == typo_at:
                continue
            ids = ids_for(w)
            cand = ids if cand is None else np.intersect1d(cand, ids, assume_unique=True)
            if len(cand) == 0:
                return wd.iloc[0:0]
        if cand is None or len(cand) > 20000:
            return wd.iloc[0:0]
        col = wd['title_f'].iloc[cand]
        ok = np.ones(len(cand), dtype=bool)
        for i, w in enumerate(ws):
            if i == typo_at:                       # this word only needs a CLOSE word in the title (herry ~ harry)
                ok &= col.map(lambda t, w=w: any(fuzz.ratio(w, x) >= 80 for x in re.findall(r'\w+', t))).to_numpy()
            else:
                end = '' if w == prefix else r'(?!\w)'
                ok &= col.str.contains(r'(?<!\w)' + re.escape(w) + end, regex=True).to_numpy()
        return wd.iloc[cand[ok]]

    found = match(words)
    if len(found) == 0 and len(words) >= 2:
        for i in range(len(words)):                # one misspelled word
            found = match(words, typo_at=i)
            if len(found):
                title_words = re.findall(r'\w+', found.iloc[0]['title_f'])
                fixed = max(title_words, key=lambda x: fuzz.ratio(words[i], x))      # "herry" -> "harry"
                reading['title_from_typo'] = ' '.join(fixed if j == i else w for j, w in enumerate(words)).title()
                break
    return found


def understand_query(original_query, sys):
    """Decide how to read the query. Returns a dict:
    kind          'prefix' | 'author' | 'title_fix' | 'words'
    query         the text to search with
    partial_word  the unfinished last word (kind 'prefix')
    author_norm / author_name / corrected / said / also_title   (kind 'author')
    title_best / title_score   closest catalog title (used for messages)"""
    q = original_query.strip()
    qf = fold(q)
    n = len(q.split())
    words_typed = re.findall(r"[\w'.]+", qf)
    one = qf if n == 1 else ''

    tb = process.extractOne(proc(q), sys['title_proc'], scorer=fuzz.token_sort_ratio)
    title_best, title_score = (sys['title_list'][tb[2]], tb[1]) if tb else (None, 0)
    out = {'kind': 'words', 'query': q, 'partial_word': None, 'title_best': title_best, 'title_score': title_score}

    typed = [w for w in re.findall(r'[a-z0-9]+', qf) if len(w) > 2 and not w.isdigit()]
    has_unknown_word = any(w not in sys['vocab'] for w in typed)          # never seen in the catalog
    has_rare_word = any(sys['word_df'].get(w, 0) < 20 for w in typed)     # seen in only a few books
    english = spell.word_usage_frequency(one) if one.isalpha() else 0
    everyday_word = n == 1 and english >= 1e-6             # "king", "love": a normal word, not a name
    looks_like_name = n <= 4 and len(q) >= 4 and not (set(q.lower().split()) & DESCRIPTION_WORDS)

    # 1. unfinished last word
    if is_unfinished(sys, words_typed):
        last = words_typed[-1]
        if n == 1 and last in sys['surname_fame']:
            # it is ALSO a real author's surname ("harr" = Jonathan Harr): show those books as typed and
            # offer the longer word as a "Did you mean" link instead of replacing what was typed
            longer = completions(sys, last, limit=50)
            # suggest the word of the most-read books ("hobb" -> hobbit, not "hobbies")
            pop = sys['work_dedup']['rating_count'].to_numpy()
            out['did_you_mean'] = max(longer, key=lambda w: (pop[docs_with(sys, w)].sum(), len(docs_with(sys, w)))) if longer else None
            out['title_prefix'] = last           # its title meaning: words starting with "harr" (Harry Potter)
        else:
            return {**out, 'kind': 'prefix', 'partial_word': last}

    # 2. one word that is a person's name: a first name OR a surname, typed right or with a typo
    #    ("rowling", "stephen" -> Stephen King, "agatha" -> Agatha Christie, "stephan"/"roling" -> did you mean)
    #    Everyday words ("king", "love") and exact book titles show titles + authors instead (step 5).
    if n == 1 and len(one) >= 3 and one.isalpha() and not everyday_word and title_score < 100:
        pick = pick_name(sys, one, allow_typo=len(one) >= 5 and english < 1e-7)
        if pick and (pick['exact'] or pick['score'] >= title_score - 2):
            authors = sys['authors_by_name'][pick['name']]
            norm = authors[0]
            corrected = not pick['exact']
            return {**out, 'kind': 'author', 'author_norm': norm, 'author_name': sys['author_display'][norm],
                    'corrected': corrected, 'said': pick['name'],
                    'also_title': title_best if corrected and title_score >= 90 else None,   # only a very close title
                    'also_authors': other_authors(sys, authors[1:], norm)}

    # 3. a full author name ("stephen king", "jk roling") or one not-everyday word that is an author
    if looks_like_name and not everyday_word:
        matches = process.extract(proc(q), sys['author_proc'], scorer=fuzz.token_sort_ratio, limit=8)
        strong = [(sys['author_list'][i], s) for _, s, i in matches if s >= 85]
        # names that contain every typed word ("garcia marquez" -> Gabriel Garcia Marquez,
        # "ronald tolkien" -> John Ronald Reuel Tolkien), when the typed last word is a known surname
        full = []
        typed_last = proc(q).split()[-1] if proc(q).split() else ''
        if n >= 2 and typed_last in sys['authors_by_name'] and title_score < 85:   # not "harry potter", "artificial intelligence"
            full = [sys['author_list'][i] for _, _, i in process.extract(
                proc(q), sys['author_proc'], scorer=fuzz.token_set_ratio, score_cutoff=100, limit=8)]
        if (strong and strong[0][1] >= title_score) or full:
            top = strong[0][1] if strong else 100
            tied = [a for a, s in strong if s >= top - 5] + full    # equally good spellings of the name
            norm = max((sys['author_norm_of'][a] for a in tied), key=lambda x: sys['author_fame'].get(x, 0))
            if n == 1 and one in sys['authors_by_name']:     # a name: its best-known author ("tolkien", "twain")
                norm = sys['authors_by_name'][one][0]
            last_typed = proc(q).split()[-1] if proc(q).split() else ''
            if n >= 2 and last_typed in sys['authors_by_name']:
                famous = sys['authors_by_name'][last_typed][0]   # best-known author with the typed surname
                fame = sys['author_fame']
                if same_person(famous, norm) or (proc(norm).split()[-1] != last_typed
                                                 and fame.get(famous, 0) >= 10 * max(fame.get(norm, 0), 1)):
                    # "ronald tolkien" -> J.R.R. Tolkien (same person); "kathleen rowling" -> J.K. Rowling,
                    # not Kathleen Laing (an exact famous surname beats a first-name-only match)
                    norm = famous
            also = []
            if n == 1 and one in sys['authors_by_name']:     # "arthur": the other well-known Arthurs too
                also = other_authors(sys, sys['authors_by_name'][one][1:], norm)
            return {**out, 'kind': 'author', 'author_norm': norm, 'author_name': sys['author_display'][norm],
                    'corrected': False, 'said': one or q, 'also_title': None, 'also_authors': also}

    # 4. a misspelled title: only when a word looks like a typo, so real words stay as typed
    #    ("introduction to physics" must not become "...Metaphysics")
    if 85 <= title_score < 100 and n <= 4 and (has_unknown_word or (has_rare_word and title_score >= 90)):
        return {**out, 'kind': 'title_fix', 'query': title_best}

    # 4b. words typed without spaces ("lordoftherings"): only when no title is close
    #     (a misspelling like "frankenstine" was already fixed by step 4, not split into "frank en stine")
    if n == 1 and len(one) >= 7 and one.isalpha() and one not in sys['vocab']:
        parts = split_joined_words(sys, one)
        if parts:
            return {**out, 'query': ' '.join(parts)}

    # 5. the words as typed, with never-seen words spell-checked
    return {**out, 'query': correct_query(q, sys['vocab'])}


# ---------- search ----------

def _search_core(query, sys, mood_filter=None, category_filter=None, top_k=24, free_only=False, online=True):

    if not query or not query.strip():
        pool = sys['work_dedup']
        suggestion = None
        if category_filter and category_filter in CATEGORY_GROUPS:
            pattern = '|'.join(CATEGORY_GROUPS[category_filter])
            pool = pool[pool['categories'].str.contains(pattern, case=False, na=False)]
            suggestion = f"Popular in {category_filter}"
        if free_only:
            pool = pool[pool['read_url'] != '']
            suggestion = "Free to read"
        if mood_filter:
            pool = pool[(pool['mood'] == mood_filter) & (pool['mood_confidence'] >= 0.6)]
            suggestion = f"Popular {mood_filter} books"
        pool = pool.sort_values(['n_ratings', 'rating_count'], ascending=False).drop_duplicates('dedup_key')
        return pool.head(top_k), suggestion

    def keep(df):
        """Apply the Free / Category / Mood filters (used by the author shortcuts below)."""
        if category_filter and category_filter in CATEGORY_GROUPS:
            df = df[df['categories'].str.contains('|'.join(CATEGORY_GROUPS[category_filter]), case=False, na=False)]
        if free_only:
            df = df[df['read_url'] != '']
        if mood_filter:
            df = df[(df['mood'] == mood_filter) & (df['mood_confidence'] >= 0.6)]
        return df

    filters_on = bool(free_only or mood_filter or (category_filter in CATEGORY_GROUPS))

    def author_empty(name):
        what = 'free books' if free_only else 'books matching your filters'
        return pd.DataFrame(columns=sys['work_dedup'].columns), f"No {what} by {name}. Try turning a filter off."

    original_query = query
    reading = understand_query(original_query, sys)

    if reading['kind'] == 'author':
        # A person. But the same words may ALSO be book titles ("sherlock", "potter", "harr", "herry potter"):
        # show both meanings, the more-read one first.
        wd, order = sys['work_dedup'], ['n_ratings', 'rating_count']
        norm, name = reading['author_norm'], reading['author_name']
        said = (reading['said'] or '').title()
        by_author = (keep(wd[wd['author_norm'] == norm]).sort_values(order, ascending=False)
                     .drop_duplicates('dedup_key').head(top_k).copy())
        titles = keep(titles_meaning(sys, original_query, reading))
        titles = titles[~titles['dedup_key'].isin(set(by_author['dedup_key']))]
        titles = titles.sort_values(order, ascending=False).drop_duplicates('dedup_key')
        others = []
        for other in reading.get('also_authors') or []:          # "stephen": Stephen Hawking, Stephen Covey...
            best = keep(wd[wd['author_norm'] == other]).sort_values(order, ascending=False).drop_duplicates('dedup_key').head(2)
            if len(best):
                others.append(best)
        if len(by_author) == 0 and len(titles) == 0 and not others and filters_on:
            return author_empty(name)

        who = (f"Did you mean {said}? Showing results for author: {name}" if reading['corrected']
               else f"Showing results for author: {name}")
        if reading.get('did_you_mean'):                  # "harr": the longer word is offered as a link
            who = f"Did you mean: {reading['did_you_mean']}"
        if len(titles) == 0 and not others:
            return by_author, who

        shown = ' '.join(re.findall(r"[\w'-]+", original_query.strip())) or original_query.strip()
        a_part = by_author.head(16 if len(titles) else top_k).copy()
        a_part['section'] = f"Books by {name}"
        parts = [a_part]
        if others:
            more = pd.concat(others).copy()
            more['section'] = f"Other authors named {said}"
            parts.append(more)
        pop_a = by_author.head(10)['rating_count'].sum()
        pop_t = titles.head(10)['rating_count'].sum() if len(titles) else 0
        if len(titles) and pop_t < 0.2 * pop_a and len(by_author) >= 8:
            titles = titles.iloc[0:0]           # "rowling": books ABOUT her are not a second meaning worth showing
        if len(titles) == 0 and not others:
            return by_author, who
        if len(titles):
            t_part = titles.head(16).copy()
            t_part['section'] = (f"Titles with “{shown}”" if not reading.get('title_from_typo')
                                 else f"Books titled “{reading['title_from_typo']}”")
            if t_part.head(10)['rating_count'].sum() > a_part.head(10)['rating_count'].sum():
                # the title meaning is read far more ("potter" -> Harry Potter): it goes first
                banner = f"Also showing books by {name} below."
                if reading.get('did_you_mean'):
                    banner = who
                return pd.concat([t_part] + parts), banner
            parts.append(t_part.head(8))
        return pd.concat(parts), who

    query = reading['query']
    partial_word = reading['partial_word']
    partial = partial_word is not None
    dym_word = reading.get('did_you_mean')

    query_was_corrected = (query.lower() != original_query.lower())

    bm25, faiss_index, model_embed = sys['bm25'], sys['faiss_index'], sys['model_embed']
    work_dedup, isbn13_by_position = sys['work_dedup'], sys['isbn13_by_position']

    bm25_query = strip_stopwords(fold(query))
    bm25_scores = bm25.get_scores(bm25_query.split())

    query_vector = model_embed.encode([query]).astype('float32')
    wide_d, wide_i = faiss_index.search(query_vector, k=1000)      # wide net: used to judge "is this book related at all"
    distances, faiss_indices = wide_d[:, :50], wide_i[:, :50]       # top 50: the ranking signal (unchanged)
    max_dist = distances[0].max() if len(distances[0]) > 0 else 1

    # cosine similarity of the nearest books (vectors are unit length, faiss gives squared L2 = 2 - 2cos)
    near_cos = {}
    for idx, dist in zip(wide_i[0], wide_d[0]):
        cos = 1 - dist / 2
        isbn = isbn13_by_position[idx]
        if cos >= RELATED_COS and isinstance(isbn, str):     # books without an ISBN can't be matched back (NaN key bug)
            near_cos[isbn] = max(cos, near_cos.get(isbn, 0))
    semantic_cos = work_dedup['isbn13'].map(near_cos).fillna(0)

    faiss_score_by_isbn = {}
    for idx, dist in zip(faiss_indices[0], distances[0]):
        isbn = isbn13_by_position[idx]
        if not isinstance(isbn, str):
            continue            # NaN ISBN: as a dict key it would give this score to all ~5,800 books without an ISBN
        sim = 1 - (dist / max_dist)
        if isbn not in faiss_score_by_isbn or sim > faiss_score_by_isbn[isbn]:
            faiss_score_by_isbn[isbn] = sim

    faiss_scores = work_dedup['isbn13'].map(faiss_score_by_isbn).fillna(0).values

    max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1
    bm25_norm = bm25_scores / max_bm25

    max_pop = work_dedup['rating_count'].max()
    pop_norm = np.log1p(work_dedup['rating_count']) / np.log1p(max_pop) if max_pop > 0 else np.zeros(len(work_dedup))

    word_count = len(query.split())
    if word_count >= 5:
        bm25_weight, faiss_weight, pop_weight = 0.15, 0.75, 0.10
    elif word_count <= 2:
        bm25_weight, faiss_weight, pop_weight = 0.40, 0.20, 0.08
    else:
        bm25_weight, faiss_weight, pop_weight = 0.35, 0.55, 0.10

    author_cues = {'by', 'author', 'writer', 'written'}
    wants_author = bool(set(query.lower().split()) & author_cues)
    q_words = [w for w in bm25_query.split() if w not in author_cues]

    # Books that have every search word somewhere (title, author or category), from the word index.
    # The exact title/author check below then only looks at these few books, not all 320k.
    candidates = None
    for w in q_words:
        parts = re.findall(r'\w+', w)
        if w == partial_word:
            ids = docs_starting_with(sys, w)
        elif parts:
            ids = docs_with(sys, parts[0])
            for x in parts[1:]:
                ids = np.intersect1d(ids, docs_with(sys, x), assume_unique=True)
        else:
            continue                                             # pure punctuation: no constraint
        candidates = ids if candidates is None else np.intersect1d(candidates, ids, assume_unique=True)

    def words_in(column):
        hit = np.zeros(len(work_dedup), dtype=bool)
        if not q_words or candidates is None or len(candidates) == 0:
            return pd.Series(hit, index=work_dedup.index)
        col = work_dedup[column + '_f'].iloc[candidates]
        if column == 'author':                                   # "jk" should hit "j.k. rowling"
            col = col + ' ' + work_dedup['author_initials'].iloc[candidates]
        ok = np.ones(len(candidates), dtype=bool)
        for w in q_words:
            end = '' if w == partial_word else r'(?!\w)'            # unfinished word: "starts with"
            ok &= col.str.contains(r'(?<!\w)' + re.escape(w) + end, regex=True).to_numpy()
        hit[candidates[ok]] = True
        return pd.Series(hit, index=work_dedup.index)

    title_hit = words_in('title')
    author_hit = words_in('author')
    primary_hit = author_hit if wants_author else title_hit

    # a book is RELATED to the query if it shares a search word, or is close in meaning.
    # Author cue words ("by", "written") don't count: "books by dickens" must not match "Squeak by Example".
    # With 2+ search words, sharing just one common word isn't enough ("history of rome" must not match every
    # Computer Science book that mentions "history"): at least two of the words must appear.
    if len(q_words) >= 2:
        words_hit = np.zeros(len(work_dedup), dtype=np.int16)
        for w in dict.fromkeys(q_words):
            parts = re.findall(r'\w+', w)
            if w == partial_word:
                ids = docs_starting_with(sys, w)
            else:
                ids = np.unique(np.concatenate([docs_with(sys, x) for x in parts])) if parts else EMPTY_IDS
            words_hit[ids] += 1
        word_related = words_hit >= 2
    else:
        word_related = (bm25.get_scores(q_words) > 0) if q_words else np.zeros(len(work_dedup), dtype=bool)
    related = pd.Series(word_related, index=work_dedup.index) | title_hit | author_hit | (semantic_cos >= RELATED_COS)

    combined_scores = (bm25_weight * bm25_norm) + (faiss_weight * faiss_scores) + (pop_weight * pop_norm) + (0.40 * primary_hit.values) + (0.10 * (work_dedup['read_url'] != '').values)
    common_words = {'a', 'the', 'book', 'about', 'story', 'something', 'like', 'that'}
    query_words = set(query.lower().split())
    looks_like_name_query = len(query.split()) <= 4 and len(query) >= 4 and not (query_words & common_words)

    results = work_dedup.copy()
    results['score'] = combined_scores

    suggestion = None
    if combined_scores.max() < 0.15:
        corrections = suggest_correction(query, work_dedup)
        if corrections:
            suggestion_text, _ = corrections[0]
            suggestion = suggestion_text

    if category_filter and category_filter in CATEGORY_GROUPS:
        keywords = CATEGORY_GROUPS[category_filter]
        pattern = '|'.join(keywords)
        results = results[results['categories'].str.contains(pattern, case=False, na=False)]

    if free_only:
        results = results[results['read_url'] != '']
        if word_count <= 4:                                   # short queries: keep books that really mention every key word
            filler = {'learn', 'learning', 'guide', 'introduction', 'intro', 'basics', 'beginner', 'beginners',
                      'free', 'online', 'study', 'how', 'best', 'good', 'easy'}
            core = [w for w in q_words if w not in filler] or q_words
            text = results['search_text'] + ' ' + results['description'].fillna('').map(fold)   # title, author, category, description
            keep = pd.Series(True, index=results.index)
            for w in core:
                end = '' if w == partial_word else r'(?!\w)'
                keep &= text.str.contains(r'(?<!\w)' + re.escape(w) + end, regex=True)
            results = results[keep]

    if mood_filter:
        results = results[(results['mood'] == mood_filter) & (results['mood_confidence'] >= 0.6)]

    if filters_on:
        # With a filter on, never pad the page with unrelated books from that category: show only related ones,
        # ranked with meaning-similarity too (the top-50 signal above rarely reaches inside a small category).
        results = results[related[results.index]].copy()
        results['score'] = results['score'] + 0.3 * semantic_cos[results.index]
        if len(results) == 0:
            return results, f"No {filter_label(category_filter, free_only)} match “{original_query.strip()}”."
    elif not related.any():
        results = results.iloc[0:0]                           # nothing related anywhere: say so instead of random books

    shown = ' '.join(q_words)
    dots = '…' if partial else ''
    TITLE_LABEL = (f'Titles starting with “{shown}…”' if partial else f'Titles with “{shown}”')   # say why a book is listed
    nice = ' '.join(w.upper() if len(w) <= 3 and not set(w) & set('aeiouy') else w.title() for w in q_words)  # jk -> JK
    AUTHOR_LABEL = f'By authors named “{nice}{dots}”'
    sectioned = word_count <= 2 and not wants_author and len(q_words) > 0

    results = results.copy()
    results['section'] = ''
    if sectioned:
        results['section'] = np.where(
            title_hit[results.index], TITLE_LABEL,
            np.where(author_hit[results.index], AUTHOR_LABEL, '')
        )

    ranked = results.sort_values('score', ascending=False).drop_duplicates('dedup_key')
    ranked = ranked.assign(rank_pop=ranked['n_ratings'] + 50 * (ranked['read_url'] != ''))
    if sectioned:
        t_all = ranked[ranked['section'] == TITLE_LABEL].sort_values(['n_ratings', 'rating_count'], ascending=False)
        a_all = ranked[ranked['section'] == AUTHOR_LABEL].sort_values(['n_ratings', 'rating_count'], ascending=False)
        # Several comparably popular people with the name ("arthur": Clarke, Conan Doyle, Miller) -> at most
        # 3 books each so all of them show. One clearly dominant person ("jk" -> J.K. Rowling) is not capped.
        person = a_all['author_norm'].map(lambda a: (proc(a).split() or ['?'])[-1] + (proc(a) or '?')[0])
        reads = a_all.groupby(person)['n_ratings'].sum().sort_values(ascending=False)
        dominant = len(reads) >= 2 and reads.iloc[0] >= 10 * max(reads.iloc[1], 1)
        if len(reads) >= 3 and not dominant:
            a_all = a_all[person.groupby(person).cumcount() < 3]
        # the reading of the query that readers rated more goes first ("tolkien" -> the author, "harry" -> the titles)
        author_first = (len(a_all) > 0 and len(t_all) > 0 and
                        a_all['n_ratings'].head(3).fillna(0).sum() > t_all['n_ratings'].head(3).fillna(0).sum())
        if len(t_all) == 0 or len(a_all) == 0:          # only one kind of match: let it fill the page
            results = pd.concat([a_all, t_all]).head(top_k)
        elif author_first:
            results = pd.concat([a_all.head(16), t_all.head(8)])
        else:
            results = pd.concat([t_all.head(16), a_all.head(8)])
        if len(results) == 0:
            results = ranked.head(top_k)
        elif len(results) < top_k:
            # few exact word matches ("learn python" -> 2 titles): add books that are close in MEANING
            more = ranked[~ranked['dedup_key'].isin(set(results['dedup_key']))]
            more = more[semantic_cos[more.index] >= RELATED_COS].head(top_k - len(results)).copy()
            if len(more):
                more['section'] = 'More books like this'
                results = pd.concat([results, more])
    else:
        results = ranked.head(top_k)

    if online and (len(results) == 0 or results['score'].max() < 0.1):
        live_results = live_lookup(query)
        if live_results:
            suggestion = suggestion or "Showing live results from Open Library"
            return pd.DataFrame(live_results), suggestion
    if len(results) == 0:
        return results, suggestion or f"No books match “{original_query.strip()}”. Check the spelling or try different words."

    # A specific title that we simply don't have (e.g. a 2021 novel): say so honestly instead of
    # presenting partial word matches ("Hail Hail Camp…") as if they were what was asked for.
    best_cos = max(near_cos.values()) if near_cos else 0
    if (not filters_on and not suggestion and len(q_words) >= 2 and not wants_author and looks_like_name_query
            and not title_hit.any() and not author_hit.any() and best_cos < 0.60):
        suggestion = f"“{original_query.strip()}” isn't in our library yet. These are the closest matches."

    # 4. start the online lookup now so it runs while covers are fetched
    live_future = None
    if (online and ENABLE_LIVE_SUPPLEMENT and len(original_query.split()) <= 4
            and not wants_author and category_filter not in CATEGORY_GROUPS and not free_only):
        live_future = live_executor.submit(live_lookup, original_query, 3, 8)

    def refresh_cover(row):
        missing = pd.isna(row['cover_url']) or row['cover_url'] == ''
        if row['source'] == 'book_crossing_only' or missing:
            if not online:
                # website: no waiting on Open Library here. The browser loads the cover by ISBN by itself
                # (and shows the title card if there is none), after the page is already on screen.
                return isbn_cover_url(row['isbn13']) or (row['cover_url'] if not missing else '')
            return get_better_cover(row['isbn13'], row['title'], row['author'], row['cover_url'])
        return row['cover_url']

    results = results.copy().reset_index(drop=True)
    rows_to_fetch = results.head(5 if online else 24)
    if online:
        with ThreadPoolExecutor(max_workers=5) as executor:
            new_covers = list(executor.map(refresh_cover, [row for _, row in rows_to_fetch.iterrows()]))
    else:
        new_covers = [refresh_cover(row) for _, row in rows_to_fetch.iterrows()]
    results.loc[:len(new_covers) - 1, 'cover_url'] = new_covers
    if query_was_corrected and not suggestion:
        suggestion = f"Searched for: {query}"

    if dym_word and (not suggestion or suggestion.startswith('Searched for')):
        suggestion = f"Did you mean: {dym_word}"            # shown as a link by the page

    # 5. add online books that are not already in the catalog (e.g. published after 2004)
    if live_future is not None:
        try:
            extra = live_future.result(timeout=6)
        except Exception as e:
            print("live supplement failed:", repr(e))
            extra = []
        extra = [e for e in extra
                 if e.get('title') and make_dedup_key(e['title'], e.get('author') or '') not in sys['dedup_keys']]
        if extra:
            extra_df = pd.DataFrame(extra)
            extra_df['section'] = LIVE_LABEL
            results = pd.concat([results, extra_df], ignore_index=True)

    return results, suggestion


cover_cache = {}


def isbn_cover_url(isbn13):
    """Open Library cover image by ISBN, loaded straight by the browser (no server wait). '' if no real ISBN."""
    isbn = str(isbn13 or '').strip()
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg" if re.fullmatch(r'\d{9}[\dX]|\d{13}', isbn) else ''


def get_better_cover(isbn13, title, author, fallback_url):
    fallback_url = fallback_url if pd.notna(fallback_url) and fallback_url != '' else ''
    cache_key = isbn13 or f"{title}|{author}"
    if cache_key in cover_cache:
        return cover_cache[cache_key]

    try:
        params = {'title': title, 'author': author, 'limit': 1}
        headers = {'User-Agent': USER_AGENT}
        response = requests.get("https://openlibrary.org/search.json", params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            docs = response.json().get('docs', [])
            if docs and docs[0].get('cover_i'):
                url = f"https://covers.openlibrary.org/b/id/{docs[0]['cover_i']}-L.jpg"
                cover_cache[cache_key] = url
                return url
    except requests.exceptions.RequestException:
        pass

    cover_cache[cache_key] = fallback_url
    return fallback_url

def top_books(sys, category_filter=None, top_k=24, m=25):
    """Best-rated books. Bayesian average: books with few ratings are pulled toward the overall mean."""
    wd = sys['work_dedup']
    pool = wd[wd['avg_rating'].notna()]
    if category_filter and category_filter in CATEGORY_GROUPS:
        pattern = '|'.join(CATEGORY_GROUPS[category_filter])
        pool = pool[pool['categories'].str.contains(pattern, case=False, na=False)]
    if pool.empty:
        return pool, "No rated books in this category yet"
    C = pool['avg_rating'].mean()
    v = pool['n_ratings']
    pool = pool.assign(top_score=(v / (v + m)) * pool['avg_rating'] + (m / (v + m)) * C)
    pool = pool.sort_values('top_score', ascending=False).drop_duplicates('dedup_key')
    note = " · ranked by average rating, weighted by number of ratings"
    label = (f"Top rated in {category_filter}" if category_filter else "Top rated books") + note
    return pool.head(top_k), label

def add_why(df, query, banner='', category_filter=None):
    """Add a 'why' column: short reasons joined by '|' (the template shows them as chips)."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    cues = {'by', 'author', 'writer', 'written'}
    q = (query or '').strip()
    words = [w for w in strip_stopwords(fold(q)).split() if w not in cues] if q else []
    author_banner = bool(banner) and banner.startswith(('Showing results for author', 'Did you mean'))

    def val(r, key):
        v = r.get(key, '')
        return '' if pd.isna(v) else v

    def whole(w, text):
        return re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', text) is not None

    all_reasons = []
    for _, r in df.iterrows():
        if val(r, 'source') == 'live_lookup':
            all_reasons.append('Found online (Open Library)')
            continue
        title_f = val(r, 'title_f') or fold(val(r, 'title'))
        author_f = val(r, 'author_f') or fold(val(r, 'author'))
        section = str(val(r, 'section'))
        why = []
        if section.startswith('Books by') or (author_banner and not section):
            why.append(f"By {val(r, 'author')}")
        elif section.startswith('Books titled'):
            why.append('Title match')
        elif not words:
            why.append(f"Popular in {category_filter}" if category_filter else 'Popular with readers')
        elif all(whole(w, title_f) for w in words):
            why.append('Title match')
        elif all(whole(w, author_f) for w in words):
            why.append('Author match')
        else:
            why.append('Similar in meaning')
        if val(r, 'read_url'):
            why.append('Free to read')
        if pd.notna(r.get('avg_rating', None)) and val(r, 'avg_rating') != '':
            why.append(f"{float(r['avg_rating']):.1f}\u2605 from {int(r['n_ratings'])} ratings")
        all_reasons.append('|'.join(why[:4]))
    df['why'] = all_reasons
    return df


# Short forms students type. Only used when the whole query is the short form, and only unambiguous ones
# ("hp" is left out: Harry Potter or H.P. Lovecraft? the initials search handles it).
SHORT_FORMS = {
    'ai': 'artificial intelligence', 'ml': 'machine learning', 'dl': 'deep learning',
    'nlp': 'natural language processing', 'cv': 'computer vision', 'dsa': 'data structures and algorithms',
    'ds': 'data structures', 'dbms': 'database management systems', 'oop': 'object oriented programming',
    'oops': 'object oriented programming', 'os': 'operating systems', 'cn': 'computer networks',
    'toc': 'theory of computation', 'coa': 'computer organization and architecture',
    'dld': 'digital logic design', 'se': 'software engineering', 'eee': 'electrical and electronic engineering',
    'cse': 'computer science and engineering', 'lotr': 'lord of the rings', 'hhgttg': "hitchhiker's guide to the galaxy",
}


def isbn_search(isbn, sys):
    """A typed ISBN (10 or 13 digits) -> that book, whichever edition the number belongs to."""
    isbn13 = to_isbn13(isbn.upper()) or isbn
    wd, books = sys['work_dedup'], sys['books']
    keys = set(books.loc[books['isbn13'] == isbn13, 'dedup_key'])
    found = wd[wd['dedup_key'].isin(keys) | (wd['isbn13'] == isbn13)].drop_duplicates('dedup_key')
    if len(found):
        return found.head(24), f"Book with ISBN {isbn}"
    return found, f"No book with ISBN {isbn} in our library."


def search_books(query, sys, mood_filter=None, category_filter=None, top_k=24, free_only=False, online=True):
    """Public entry point: tidy the query, expand short forms, run the search, then attach the 'why' chips."""
    query = re.sub(r'(?<=[^\W\d_])[-_/]+(?=[^\W\d_])', ' ', query or '')   # "harry-potter" -> "harry potter"
    digits = re.sub(r'[\s-]', '', query)
    if re.fullmatch(r'\d{9}[\dXx]|\d{13}', digits):                          # an ISBN
        df, banner = isbn_search(digits, sys)
        return add_why(df, query, banner, category_filter), banner
    short = fold(query or '').strip().strip('.')
    expanded = SHORT_FORMS.get(short)
    q = expanded or query
    df, banner = _search_core(q, sys, mood_filter=mood_filter, category_filter=category_filter,
                              top_k=top_k, free_only=free_only, online=online)
    # Safety net: nothing found, but a title is a very close spelling match -> search that title instead
    if len(df) == 0 and q and q.strip() and (banner or '').startswith('No books match'):
        tb = process.extractOne(proc(q), sys['title_proc'], scorer=fuzz.token_sort_ratio, score_cutoff=88)
        if tb:
            title = sys['title_list'][tb[2]]
            df2, _ = _search_core(title, sys, mood_filter=mood_filter, category_filter=category_filter,
                                  top_k=top_k, free_only=free_only, online=online)
            if len(df2):
                df, banner, q = df2, f"Did you mean “{title}”? Showing results for it.", title
    if expanded and (not banner or banner.startswith('Searched for')):
        banner = f"Showing results for “{expanded}” ({short.upper()})"
    return add_why(df, q, banner, category_filter), banner

def todays_pick(sys, day=None):
    import datetime
    import hashlib
    wd = sys['work_dedup']
    if 'read_url' not in wd.columns:
        return None
    free = wd[wd['read_url'].fillna('') != ''].copy()
    if len(free) == 0:
        return None
    free['_base'] = free['title_f'].str.replace(r'\s*[\(\[].*$', '', regex=True).str.strip() + '|' + free['author_f']
    free = free.sort_values('n_ratings', ascending=False).drop_duplicates('_base')   # keep the best-known edition
    free['_h'] = [hashlib.md5(str(k).encode()).hexdigest() for k in free['dedup_key']]
    pool = free.sort_values('_h')                      # fixed shuffled order: no alphabetical or popularity pattern
    day = day or datetime.date.today()
    row = pool.iloc[day.toordinal() % len(pool)]
    one = add_why(row.drop(['_base', '_h']).to_frame().T.reset_index(drop=True), '', '', None)
    return one.iloc[0].to_dict()

def online_extra(query, sys, limit=6):
    """Books from Open Library that are NOT already in our catalog (e.g. published after 2004).
    The results page loads these after it is already on screen, so a slow Open Library never delays a search."""
    q = (query or '').strip()
    if not q:
        return pd.DataFrame()
    found = live_lookup(q, limit, 8)
    keep = [e for e in found
            if e.get('title') and make_dedup_key(e['title'], e.get('author') or '') not in sys['dedup_keys']]
    if not keep:
        return pd.DataFrame()
    df = pd.DataFrame(keep)
    df['section'] = LIVE_LABEL
    return add_why(df, q, '', None)
