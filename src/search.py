import pandas as pd
import numpy as np
import os
import re
import unicodedata
import bisect
import math
import heapq
import itertools
import faiss
from rapidfuzz import fuzz, process, utils
from rapidfuzz.distance import DamerauLevenshtein
from sentence_transformers import SentenceTransformer
import requests
import html
from spellchecker import SpellChecker
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


spell = SpellChecker()
spell_quick = SpellChecker(distance=1)   # the last-resort correction: 1 wrong letter (2 is very slow)

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
        c = spell_quick.correction(w)
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
    'Science': [r'(?<!social )(?<!political )(?<!library )(?<!computer )science(?![\s-]*fiction)',
                'physics', 'biology', 'chemistry'],   # not "Science-Fiction", "Social Science"
}
ACADEMIC_GROUPS = {'Computer Science', 'Mathematics', 'Engineering', 'Science'}
# a novel tagged "Computers" or "Science" (All Systems Red, Dungeon Crawler Carl) is not a CS or science book
NOT_ACADEMIC = r'(?<!non)(?<!non-)(?<!non )fiction|\bnovels?\b|juvenile|graphic novel|comics|fantasy'


def in_category(df, group):
    """Rows of `df` that belong to a category group (the one place category matching is decided)."""
    if not group or group not in CATEGORY_GROUPS:
        return df
    cats = df['categories'].fillna('')
    mask = cats.str.contains('|'.join(CATEGORY_GROUPS[group]), case=False, regex=True)
    if group in ACADEMIC_GROUPS:
        mask &= ~cats.str.contains(NOT_ACADEMIC, case=False, regex=True)
    return df[mask]

# words that make a query a study subject (free textbooks first, novels that share the word last)
STUDY_WORDS = (r'\b(calculus|algebra|geometry|trigonometry|statistics|probability|mathematics|math|physics|'
               r'chemistry|biology|economics|accounting|programming|python|java|javascript|algorithms?|'
               r'database|networking|engineering|anatomy|physiology|psychology|sociology|microeconomics|'
               r'macroeconomics|thermodynamics|mechanics|electronics|circuits|machine learning)\b')

# keys next to each other on a QWERTY keyboard: a wrong FIRST letter is usually one of these
KEY_NEIGHBOURS = {'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tugh', 'u': 'yihj',
                  'i': 'uojk', 'o': 'ipkl', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
                  'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop', 'z': 'asx', 'x': 'zsdc',
                  'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'}

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


class FastBM25:
    """The same scores as rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25), but each query word only
    touches the books that contain it (an index built once) instead of looping over all 500,000 books in
    Python on every search: ~0.35 s -> a few ms. Also uses far less memory than 500,000 small dicts."""

    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.doc_len = np.fromiter((len(d) for d in corpus), dtype=np.float32, count=self.n)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        ids, tfs = {}, {}
        for i, doc in enumerate(corpus):
            for w, c in Counter(doc).items():
                ids.setdefault(w, []).append(i)
                tfs.setdefault(w, []).append(c)
        self.postings = {w: (np.asarray(ids[w], dtype=np.int32), np.asarray(tfs[w], dtype=np.float32)) for w in ids}
        del ids, tfs
        idf, negative, total = {}, [], 0.0
        for w, (d, _) in self.postings.items():
            v = math.log(self.n - len(d) + 0.5) - math.log(len(d) + 0.5)
            idf[w] = v
            total += v
            if v < 0:
                negative.append(w)
        eps = epsilon * (total / len(idf) if idf else 0)
        for w in negative:
            idf[w] = eps
        self.idf = idf
        self.norm = (self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)).astype(np.float32)

    def get_scores(self, query):
        score = np.zeros(self.n, dtype=np.float64)
        for q in query:
            hit = self.postings.get(q)
            if hit is None:
                continue
            d, tf = hit
            score[d] += self.idf[q] * (tf * (self.k1 + 1) / (tf + self.norm[d]))
        return score


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
        # Open Library's popularity is "people who logged the book" (Atomic Habits: 66,000), about 20x the
        # Book-Crossing rating counts (The Lovely Bones: 1,300). Put both on one scale so neither dominates.
        is_ol = work_dedup['source'].eq('import_ol')
        work_dedup.loc[is_ol, 'rating_count'] = (pd.to_numeric(work_dedup.loc[is_ol, 'rating_count'], errors='coerce')
                                                 .fillna(0) / 20).round()
        new_keys = set(ex_new['dedup_key'])
        linked_old = sum(1 for k in extra_read if k not in new_keys)
        print(f"Extra books added: {len(ex_new)} (free links added to {linked_old} books already in the catalog)")

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

    # newer books (2005+) imported from Open Library bring Open Library's own reader ratings
    if 'ol_count' in work_dedup.columns:
        ol_n = pd.to_numeric(work_dedup['ol_count'], errors='coerce').fillna(0)
        ol_avg = pd.to_numeric(work_dedup['ol_avg'], errors='coerce')
        use = work_dedup['avg_rating'].isna() & (ol_n >= 5) & ol_avg.notna()
        work_dedup.loc[use, 'avg_rating'] = ol_avg[use]
        work_dedup.loc[use, 'n_ratings'] = ol_n[use].astype(int)

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
    bm25 = FastBM25(tokenized_corpus)
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
    # word -> titles that contain it (lets the closest-title check skip unrelated titles)
    tw_lists = {}
    for i, t in enumerate(title_proc):
        for w in set(t.split()):
            if len(w) >= 3:
                tw_lists.setdefault(w, []).append(i)
    title_word_ids = {w: np.asarray(v, dtype=np.int32) for w, v in tw_lists.items()}
    title_words = sorted(title_word_ids)
    del tw_lists
    aw_lists = {}
    for i, a in enumerate(author_proc):
        for w in set(a.split()):
            if len(w) >= 3:
                aw_lists.setdefault(w, []).append(i)
    author_word_ids = {w: np.asarray(v, dtype=np.int32) for w, v in aw_lists.items()}
    author_words = sorted(author_word_ids)
    del aw_lists

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
    name_set = set(name_words)
    fix_words = [w for w in index_words if len(w) >= 3 and w.isalpha() and (len(word_index[w]) >= 3 or w in name_set)]
    author_display = ranked_authors.drop_duplicates('author_norm').set_index('author_norm')['author'].to_dict()
    author_norm_of = work_dedup.drop_duplicates('author').set_index('author')['author_norm'].to_dict()

    # numbers that are the same for every search: computed once here, not on each search
    rc = work_dedup['rating_count'].fillna(0).to_numpy(dtype=np.float64)
    pop_norm_all = np.log1p(rc) / np.log1p(rc.max()) if rc.max() > 0 else np.zeros(len(work_dedup))
    has_read_all = (work_dedup['read_url'] != '').to_numpy() if 'read_url' in work_dedup.columns         else np.zeros(len(work_dedup), dtype=bool)
    rows_by_isbn = {}                                   # isbn -> row numbers in work_dedup (for meaning scores)
    for i, isbn in enumerate(work_dedup['isbn13'].tolist()):
        if isinstance(isbn, str):
            rows_by_isbn.setdefault(isbn, []).append(i)
    rows_by_isbn = {k: np.asarray(v, dtype=np.int64) for k, v in rows_by_isbn.items()}

    return {
        'pop_norm': pop_norm_all, 'has_read': has_read_all, 'rows_by_isbn': rows_by_isbn,
        'title_word_ids': title_word_ids, 'title_words': title_words,
        'author_word_ids': author_word_ids, 'author_words': author_words,
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
        'authors_by_name': authors_by_name, 'name_words': name_words, 'fix_words': fix_words,
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


def word_forms(w):
    """The same word with simple endings: advance/advanced, algorithm/algorithms, process/processes."""
    forms = {w, w + 's', w + 'es', w + 'd', w + 'ed'}
    if len(w) > 4 and w.endswith('es'):
        forms.add(w[:-2])
    if len(w) > 3 and w.endswith('s'):
        forms.add(w[:-1])
    if len(w) > 4 and w.endswith('ed'):
        forms |= {w[:-1], w[:-2]}
    return forms


def docs_with_forms(sys, w):
    """Books with the word in any of its simple forms."""
    parts = [docs_with(sys, f) for f in word_forms(w)]
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else EMPTY_IDS


def form_pattern(w):
    """Regex for the word in any simple form, as a whole word."""
    return r'(?<!\w)(?:' + '|'.join(re.escape(f) for f in sorted(word_forms(w), key=len, reverse=True)) + r')(?!\w)'


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
        # similar names only when they are comparably known (Sedgwick ~ Sedgewick), not noise under Rowling
        close = [nm for nm, _ in near if DamerauLevenshtein.distance(word, nm) == 1
                 and fame_of(nm) >= max(1, 0.1 * fame_of(word))]
        return {'name': word, 'score': 100, 'exact': True, 'others': close[:2]}
    # a word our books really contain (Jurafsky, listed as a co-author) is not a misspelled name
    if allow_typo and not len(docs_with(sys, word)):
        limit_edits = 1 if len(word) <= 5 else 2      # short words: 1 wrong letter ("frodo" is not Fyodor)
        wide = []
        for nm, sc, _ in process.extract(word, sys['name_words'], scorer=fuzz.ratio, score_cutoff=72, limit=20):
            edits = DamerauLevenshtein.distance(word, nm)
            if nm != word and edits <= limit_edits and nm[0] == word[0]:
                wide.append((nm, sc, edits))       # a real catalog word keeps its first letter ("frodo" is not Brodo)
        if word in sys['vocab']:                   # a word our books contain: only very close, much better-known names
            wide = [x for x in wide if x[1] >= 85]
        if wide:
            # fewer wrong letters is likelier, but a far better-known name wins ("rolimg" -> rowling, not rolim)
            rank = lambda x: -30 * x[2] + 10 * math.log10(fame_of(x[0]) + 1)
            wide.sort(key=rank, reverse=True)
            nm, sc, _ = wide[0]
            # the other close names are other meanings ("hawkings": Hawkins AND Hawking), shown as sections
            others = [x[0] for x in wide[1:] if x[2] <= wide[0][2] and fame_of(x[0]) >= 5][:3]
            # a well-known name may be 2 letters off and still win over a so-so title match
            return {'name': nm, 'score': max(sc, 85) if fame_of(nm) >= 1000 else sc, 'exact': False, 'others': others}
    return None


def best_title_word(sys, word):
    """The catalog word a misspelled word most likely means AS A TITLE WORD: close spelling, much-read books."""
    pop = sys['work_dedup']['rating_count'].fillna(0).to_numpy()
    max_edits = 1 if len(word) <= 5 else 2 if len(word) <= 6 else 3
    best, best_score = None, -1e9
    for cand, _, _ in process.extract(word, sys['fix_words'], scorer=fuzz.ratio, score_cutoff=70, limit=40):
        edits = DamerauLevenshtein.distance(word, cand)
        if cand == word or edits > max_edits or cand[0] != word[0] or cand in STOPWORDS:
            continue
        score = -30 * edits + 6 * math.log10(1 + pop[docs_with(sys, cand)].sum())
        if score > best_score:
            best, best_score = cand, score
    return best


def titles_meaning(sys, original_query, reading):
    """Books whose TITLE contains the typed words: the other meaning of a name-like query.
    The last word may be unfinished ("harr" -> Harry...), and in a multi-word query one word may be
    misspelled ("herry potter" -> Harry Potter)."""
    wd = sys['work_dedup']
    words = [w for w in re.findall(r'\w+', fold(original_query)) if w not in STOPWORDS and len(w) >= 2]
    if reading.get('title_word'):                     # one misspelled word: its title meaning (sherlok -> Sherlock)
        words = [reading['title_word']]
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
                ok &= col.map(lambda t, w=w: any(fuzz.ratio(w, x) >= (75 if len(w) <= 5 else 80) for x in re.findall(r'\w+', t))).to_numpy()
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


def fix_typos(sys, query):
    """Fix misspelled words using the words our books really contain, choosing the spelling that fits the
    OTHER words typed: "lord of the rigns" -> rings (books have "lord" + "rings"), "percy jakson" -> jackson,
    "jk rolimg" -> rowling (a name, next to initials), "chilfrend's" -> children's, "twilite" -> twilight.
    Only words our catalog and the English dictionary have never seen are touched. -> the fixed query."""
    tokens = query.split()
    folded = [fold(t).strip('.,!?;:"()') for t in tokens]
    bases = [f[:-2] if f.endswith(("'s", "’s")) else f for f in folded]
    known_ids = [docs_with(sys, b) for b in bases
                 if b in sys['vocab'] and b not in STOPWORDS and len(b) >= 3 and len(docs_with(sys, b))]
    name_context = any(b not in STOPWORDS and b in sys['authors_by_name'] or (len(b.replace('.', '')) <= 3 and b.replace('.', '').isalpha()
                                                        and b not in STOPWORDS) for b in bases)
    pop = sys['work_dedup']['rating_count'].fillna(0).to_numpy()
    out, changed = [], False
    for tok, f, b in zip(tokens, folded, bases):
        # only words NO book has (a word in even one book, like Jurafsky, is real)
        if not (len(b) >= 4 and b.isalpha() and not len(docs_with(sys, b)) and not spell.known([b])):
            out.append(tok)
            continue
        best, best_score = None, -1e9
        max_edits = 1 if len(b) <= 5 else 2 if len(b) <= 6 else 3     # "twilite" -> twilight is 3
        for cand, ratio, _ in process.extract(b, sys['fix_words'], scorer=fuzz.ratio, score_cutoff=70, limit=40):
            edits = DamerauLevenshtein.distance(b, cand)   # a wrong, missing, extra or swapped letter = 1 edit
            if edits > max_edits or cand[0] != b[0]:          # the first letter is kept ("frodo" is not Brodo)
                continue
            ids = docs_with(sys, cand)
            score = -30 * edits + 6 * math.log10(len(ids) + 1) + 3 * math.log10(1 + pop[ids].sum())   # common, much-read words
            together = [np.intersect1d(ids, k, assume_unique=True) for k in known_ids]
            if together and all(len(t) for t in together):
                # it appears in books together with the other words: the more those books are read, the
                # likelier ("animal farn" -> farm: Animal Farm is read far more than Animal Fair)
                both = together[0]
                for t in together[1:]:
                    both = np.intersect1d(both, t, assume_unique=True)
                score += 15 + 4 * math.log10(1 + pop[both].sum() if len(both) else 1)
            if name_context and cand in sys['authors_by_name']:
                score += 6
            if score > best_score:
                best, best_score = cand, score
        if best:
            changed = True
            out.append(best + (f[len(b):] if f != b else ''))
        else:
            out.append(tok)
    return ' '.join(out) if changed else query


def _max_edits(word):
    return 1 if len(word) <= 5 else 2 if len(word) <= 6 else 3     # "twilite" -> twilight is 3


SOUND_RULES = [('tch', 'ch'), ('ght', 't'), ('igh', 'i'), ('ite', 'it'), ('ph', 'f'), ('ck', 'k'), ('qu', 'kw'),
               ('x', 'ks'), ('wr', 'r'), ('kn', 'n'), ('ce', 'se'), ('ci', 'si'), ('cy', 'sy'), ('c', 'k'),
               ('z', 's'), ('y', 'i'), ('ee', 'i'), ('ea', 'i'), ('oo', 'u'), ('ou', 'u')]


def sounds_like(word):
    """A rough 'how it sounds' key, so spellings that sound the same match even when many letters differ:
    twilite / twilight -> 'twilit', fonix / phonix -> 'fonics', sherlok / sherlock -> 'sherlok'."""
    w = re.sub(r'[^a-z]', '', fold(word))
    for a, b in SOUND_RULES:
        w = w.replace(a, b)
    w = re.sub(r'(.)\1+', r'\1', w)                  # double letters sound like one
    w = w.translate(VOICED_TO_UNVOICED)              # d/t, b/p, g/k, v/f are said almost alike: poder = potter
    w = re.sub(r'(.)\1+', r'\1', w)
    return w[:-1] if len(w) > 3 and w.endswith('e') else w   # silent final e


VOICED_TO_UNVOICED = str.maketrans('bdgv', 'ptkf')
# pairs that are mixed up as the FIRST letter ("dolkien" = Tolkien, "brejudice" = prejudice, "kadsby" = Gatsby)
FIRST_LETTER_TWINS = {'b': 'p', 'p': 'b', 'd': 't', 't': 'd', 'g': 'k', 'k': 'gc', 'c': 'k', 'v': 'f', 'f': 'v', 's': 'z', 'z': 's'}


def warm_up_indexes(sys):
    """Build the one-off helper indexes (about 25 seconds in all) in the background at start-up, so the first
    visitor who makes a heavy typo does not wait for them."""
    for build in (lambda: popular_phrase_index(sys), lambda: same_sound_words(sys, 'harry'),
                  lambda: _sound_index(sys, 'author'), lambda: _sound_index(sys, 'title')):
        try:
            build()
        except Exception as e:                      # never let warm-up break the app
            print('index warm-up failed:', repr(e))


def books_with_all_words(sys, q):
    """How many books contain every meaningful word of `q` (0 = a reading no book has, such as 'agatha hartini')."""
    ids = None
    for t in q.split():
        w = fold(t).strip('.,!?;:"()')
        if w.isalpha() and len(w) >= 3 and w not in STOPWORDS:
            d = docs_with(sys, w)
            ids = d if ids is None else np.intersect1d(ids, d)
    return 0 if ids is None else len(ids)


def same_sound_words(sys, word):
    """Catalog words that SOUND like `word` (hari = harry, twilite = twilight), however different they look.
    The index is built once, on first use."""
    index = sys.get('sound_index')
    if index is None:
        index = {}
        for w in sys['fix_words']:
            index.setdefault(sounds_like(w), []).append(w)
        sys['sound_index'] = index
    return index.get(sounds_like(word), [])


def spelling_alternatives(sys, query, keep=3, margin=15, slips=False):
    """Every likely meaning of a query that contains misspelled words, most likely first.

    A misspelled word is often close to SEVERAL real words: "poter" -> potter AND peter,
    "sherlok" -> sherlock, "harr poter" -> harry potter. Instead of picking one fix per word,
    this lists every catalog word 1-3 letters away from each misspelled word, then scores each
    whole combination by how close it is (fewer wrong letters) and how much the books that
    contain ALL the words are read. Returns a list of query strings (best first), or [] when
    nothing in the query looks misspelled. The original word stays in the list when the catalog
    really has it (e.g. "harr" = Jonathan Harr), so both meanings can be shown.
    """
    tokens = query.split()
    if not 1 <= len(tokens) <= 5:
        return []
    pop = sys['work_dedup']['rating_count'].fillna(0).to_numpy()

    def pop_of(ids):
        """How much the 10 most-read books with these words are read. Not the total: "peter" is in
        2,800 books, "potter" in 366, but Harry Potter is read far more than any Peter book."""
        if not len(ids):
            return 0.0
        p = pop[ids]
        return float(np.partition(p, len(p) - 10)[-10:].sum()) if len(p) > 10 else float(p.sum())

    slots = []             # per token: list of (word, edits)
    slipped = set()        # words reached only through a wrong FIRST key ("orcy" -> percy)
    pop_of_reading = {}
    known_ids = []         # books for the correctly spelled, meaningful words
    typed_real = []        # misspelled-looking words that ARE in our books (surnames: "podder", "harr")
    known_by_slot = {}     # slot index -> books for a correctly spelled word that could still be a slip
    real_words = []        # (slot index, word, token, books) of those words
    any_typo = False
    for tok in tokens:
        b = fold(tok).strip('.,!?;:"()')
        plain = b.isalpha() and len(b) >= 4 and b not in STOPWORDS
        if not plain:
            slots.append([(tok, 0)])
            if b.isalpha() and b not in STOPWORDS and len(b) >= 3 and len(docs_with(sys, b)):
                known_ids.append(docs_with(sys, b))
            continue
        own_ids = docs_with(sys, b)
        cands = []
        # what matters is whether OUR BOOKS have the word: "dostoevski" is in the English dictionary,
        # but the catalog spells it Dostoevsky
        maybe_typo = b not in sys['vocab'] or (len(own_ids) < 20 and not spell.known([b]))
        key = sounds_like(b)
        pool = []
        sound_twins = set()    # words reached by swapping a doubled consonant for its twin sound
        if maybe_typo:
            pool = [c for c, _, _ in process.extract(b, sys['fix_words'], scorer=fuzz.ratio, score_cutoff=70, limit=60)]
            pool += same_sound_words(sys, b)          # "poder" does not LOOK like "potter", but sounds the same
            # a doubled consonant swapped for its twin sound: "podder" = potter, "hobbit" typed "hoppit"
            for old, new in (('dd', 'tt'), ('tt', 'dd'), ('bb', 'pp'), ('pp', 'bb'), ('gg', 'kk'), ('kk', 'gg')):
                if old in b:
                    twin = b.replace(old, new)
                    extra = [c for c, _, _ in process.extract(twin, sys['fix_words'], scorer=fuzz.ratio,
                                                              score_cutoff=90, limit=5)]
                    sound_twins.update(extra)
                    pool += extra
            for k in (KEY_NEIGHBOURS.get(b[0], '') if slips else ''):   # wrong first key: "orcy" -> "prcy" -> percy
                pool += [c for c, _, _ in process.extract(k + b[1:], sys['fix_words'], scorer=fuzz.ratio,
                                                          score_cutoff=80, limit=10)]
        for cand in dict.fromkeys(pool):
            if cand == b or cand in STOPWORDS:
                continue
            neighbour = (slips and cand[0] != b[0] and b[0] in KEY_NEIGHBOURS.get(cand[0], '')
                         and DamerauLevenshtein.distance(b[1:], cand[1:]) <= _max_edits(b))
            twin_start = (cand[0] in FIRST_LETTER_TWINS.get(b[0], '') and sounds_like(cand) == key)
            swapped_start = cand == b[1:2] + b[:1] + b[2:]         # first two letters swapped: "uasten" = austen
            vowel_start = cand[1:] == b[1:] and cand[0] in 'aeiou' and b[0] in 'aeiou'    # "eperating" = operating
            if cand[0] != b[0] and cand[1:] != b and not neighbour and not twin_start and not swapped_start and not vowel_start:
                continue          # first letter kept, unless left out, a key next to it was hit ("orcy" = percy), or a twin sound
            edits = DamerauLevenshtein.distance(b, cand)
            # a word that SOUNDS the same may be 2 letters away even when it is short ("poder" = potter)
            sound_ok = edits <= 2 and sounds_like(cand) == key and (cand[:2] == b[:2] or twin_start)
            if edits <= _max_edits(b) or neighbour or sound_ok:
                # cost in "wrong letters", plus a little for how different the words look overall, so
                # equally popular choices go to the closer spelling; a word that SOUNDS the same counts
                # as much closer ("twilite" sounds like twilight, not twisted)
                cost = edits + 0.012 * (100 - fuzz.ratio(b, cand))
                if edits >= 2 and sounds_like(cand) == key:      # only for big typos: 1-letter typos are
                    cost = max(0.5, cost - 1.5)                  # fine already ("christe" is Christie, not Christ)
                if len(b) >= 5 and cand.startswith(b) and len(cand) - len(b) <= 3:
                    cost = max(0.4, cost - 0.7)                  # half-typed word: "stephe" is stephen, not steve
                if edits > _max_edits(b) and not neighbour:
                    cost = max(cost, 1.2)                        # let in only for its sound: just below an equal 1-letter fix
                if neighbour:                                    # one slipped key + the rest of the word
                    cost = DamerauLevenshtein.distance(b[1:], cand[1:]) + 0.5
                    slipped.add(cand)
                cands.append((cand, cost, edits))
        # a slipped first key is believed only when it is CLEARLY closer than any word that keeps the first letter:
        # "fatsby" = gatsby (1 letter, against 2 for "fats"), but "kunth" is knuth (1), not "month"
        best_plain = min((c[2] for c in cands if c[0] not in slipped), default=99)
        cands = [c for c in cands if c[0] not in slipped or c[2] < best_plain]
        # never seen in our books; a real English word is only fixed when a catalog word is 1 letter away
        # (a word in even ONE of our books is real: Sipser, Mankiw, Jurafsky have 1-2 books)
        unknown = not len(own_ids) and (not spell.known([b]) or any(c[2] == 1 for c in cands))
        # a rare word our books do contain is still treated as a likely typo when a close word is read
        # FAR more ("harr" -> harry, "tolkin" -> tolkien); the word as typed stays as another meaning
        own_pop = pop_of(own_ids)
        much_better = [c for c in cands if pop_of(docs_with(sys, c[0])) >= 10 * max(own_pop, 1)
                       and pop_of(docs_with(sys, c[0])) >= 200
                       and (c[2] == 1 or not len(own_ids) or c[0] in sound_twins
                            or (c[2] == 2 and sounds_like(c[0]) == key))]   # a word our books have: only 1 letter off ("hermione" stays), or the same sound ("podder", "poder" = potter)
        sfame = sys['surname_fame']
        if b in sfame and len(tokens) == 1:
            # (only for a one-word query: in "harr podder" the OTHER words decide, so "harry" stays a candidate)
            # typed exactly an author's surname ("cormen", "sipser", "mankiw"): it is only a typo when a far
            # better-known SURNAME is that close ("tolkin" -> Tolkien), never a common word ("carmen", "sister")
            much_better = [c for c in much_better if sfame.get(c[0], 0) >= 10 * max(sfame.get(b, 0), 1)]
        if (unknown and cands) or (len(own_ids) < 20 and not spell.known([b]) and much_better):
            any_typo = True
            ranked = sorted(((c[0], c[1]) for c in cands),
                            key=lambda c: -25 * c[1] + 10 * math.log10(1 + pop_of(docs_with(sys, c[0]))),
                            reverse=True)[:4]
            best_pop = max((pop_of(docs_with(sys, c[0])) for c in ranked), default=0)
            if len(own_ids):
                ranked.append((b, 1))              # as typed: a real but much less-read meaning ("harr", "podder")
                typed_real.append(b)
            slots.append(ranked)
            if len(own_ids):
                real_words.append((len(slots) - 1, b, tok, own_ids))
        else:
            slots.append([(tok, 0)])
            if len(own_ids) and b not in STOPWORDS:
                known_by_slot[len(slots) - 1] = own_ids
                real_words.append((len(slots) - 1, b, tok, own_ids))

    # a REAL word of a longer query can still be a slip for a far more read one that sounds the same:
    # "hari poter" / "hari potter" -> "harry potter" ("hari" is Mata Hari, Hari Kunzru, but 2 letters from harry).
    # The swapped word is judged together with the others below, so only a reading that makes books wins.
    extended = set()
    if len(tokens) >= 2:
        for idx, b, tok, own_ids in real_words:
            own_pop = pop_of(own_ids)
            extra = []
            # "hari" -> "harry" looks only 67% alike (many other words look closer) but SOUNDS the same: so words that
            # sound the same are candidates too
            pool_words = [c for c, _, _ in process.extract(b, sys['fix_words'], scorer=fuzz.ratio, score_cutoff=70, limit=40)]
            for cand in dict.fromkeys(pool_words + same_sound_words(sys, b)):
                if cand == b or cand in STOPWORDS:
                    continue
                # the first letter: kept, left out ("ing" = king), a key next to it ("garry" = harry), a twin sound
                # ("dolkien"), or the first two letters swapped ("uasten" = austen)
                first_ok = (cand[0] == b[0] or cand[1:] == b or b[0] in KEY_NEIGHBOURS.get(cand[0], '')
                            or cand[0] in FIRST_LETTER_TWINS.get(b[0], '') or cand == b[1:2] + b[:1] + b[2:]
                            or (cand[1:] == b[1:] and cand[0] in 'aeiou' and b[0] in 'aeiou'))
                if not first_ok:
                    continue
                edits = DamerauLevenshtein.distance(b, cand)
                same_sound = sounds_like(cand) == sounds_like(b)
                if edits > (2 if same_sound else _max_edits(b)) or (edits >= 2 and cand[:2] != b[:2] and not same_sound):
                    continue
                cand_pop = pop_of(docs_with(sys, cand))
                if cand_pop >= 3 * max(own_pop, 1) and cand_pop >= 200:
                    cost = edits + 0.012 * (100 - fuzz.ratio(b, cand))
                    if edits >= 2 and same_sound:
                        cost = max(0.5, cost - 1.5)
                    extra.append((cand, cost, cand_pop))
            if extra:
                extra.sort(key=lambda c: -25 * c[1] + 10 * math.log10(1 + c[2]), reverse=True)
                have = {w for w, _ in slots[idx]}
                slots[idx] = slots[idx] + [(c, cost) for c, cost, _ in extra[:3] if c not in have]
                extended.add(idx)
    if not any_typo and not extended:
        return []
    known_ids += [ids for i, ids in known_by_slot.items() if i not in extended]   # a word that may be a slip must not narrow the books

    base = None
    for ids in known_ids:
        base = ids if base is None else np.intersect1d(base, ids, assume_unique=True)
    scored = []
    while math.prod(len(sl) for sl in slots) > 200:         # trim the longest list of choices (worst first)
        longest = max(range(len(slots)), key=lambda i: len(slots[i]))
        slots[longest] = slots[longest][:-1]
    loose = []
    for combo in itertools.product(*slots):
        ids = base
        edits = 0
        for word, e in combo:
            edits += e
            w = fold(word).strip('.,!?;:"()')
            if w.isalpha() and w not in STOPWORDS and len(w) >= 3:
                d = docs_with(sys, w)
                ids = d if ids is None else np.intersect1d(ids, d, assume_unique=True)
        together = pop_of(ids) if ids is not None else 0.0
        last = fold(combo[-1][0]).strip('.,!?;:"()')
        # a full name: "kathleen rowling" is J.K. Rowling though no book has both words, so a famous
        # surname counts like the books it stands for
        name_pop = 0.0
        if len(combo) >= 2 and last in sys['authors_by_name']:
            words_set = {fold(w).strip('.,!?;:"()') for w, _ in combo}
            fame = sys['author_fame']
            exact_person = [fame.get(a, 0) for a in sys['authors_by_name'][last]
                            if words_set <= set(proc(sys['author_display'].get(a, a)).split())]
            # "thomas cormen" is a real person (Thomas H. Cormen); "thomas corwin" is only a surname
            name_pop = float(max(exact_person)) * 10 if exact_person else 0.1 * float(sys['surname_fame'].get(last, 0))
        if ids is not None and len(ids) == 0 and name_pop < 200:
            # these words never appear together in one book ("calculus" + "engineers"): kept as a weak,
            # word-by-word reading, used only when NO reading has all its words together
            alone = sum(math.log10(1 + pop_of(docs_with(sys, fold(w).strip('.,!?;:"()')))) for w, e in combo if e)
            loose.append((-25 * edits + 2 * alone, ' '.join(w for w, _ in combo)))
            continue
        score = -25 * edits + 10 * math.log10(1 + max(together, name_pop))
        reading = ' '.join(w for w, _ in combo)
        pop_of_reading[reading] = together              # books that really contain all these words
        scored.append((score, reading))
    if not scored:
        if not loose:
            return []
        scored = loose                            # every word fixed on its own ("calculas for engeneers")
    scored.sort(reverse=True)
    best = scored[0][0]
    out = []
    for score, q in scored:
        if score >= best - margin and q not in out:
            out.append(q)
        if len(out) == keep:
            break
    # a reading where a misspelled word is an author's SURNAME stays as a choice even when a common word
    # scores higher ("cormon" -> common, but the student may mean Cormen): best such reading, once
    sfame = sys['surname_fame']
    for score, q in scored:
        if q not in out and any(fold(w) in sfame and fold(w) != fold(t) for w, t in zip(q.split(), tokens)):
            out.append(q)
            break
    # a slipped-first-key reading is offered only when it wins, or is read far more ("hary poter" is not
    # also "gary power"; "orcy jackso" is Percy Jackson)
    if out:
        top = pop_of_reading.get(out[0], 0)
        out = [q for i, q in enumerate(out)
               if i == 0 or not any(fold(w) in slipped for w in q.split()) or pop_of_reading.get(q, 0) >= 3 * max(top, 1)]
    # what was typed, when the catalog really has it ("harr" = Jonathan Harr), always stays as a choice
    if query not in out and any(q == query for _, q in scored):
        out.append(query)
    if out == [query]:
        return []
    out = out[:4]
    # a typed word that really is an author's surname stays one click away on its own ("podder" ->
    # Tanushree Podder), next to the likelier reading ("harry potter")
    sfame = sys['surname_fame']
    for w in typed_real:
        if w in sfame and w not in out and not any(w in q.split() for q in out):
            out.append(w)
    return out[:5]


def _mention_index(sys):
    """Lazily built: row ids of books that have a description, and those descriptions (folded, lowercase),
    kept as an Arrow array so a search over all of them runs in fast native code."""
    if 'desc_ids' not in sys:
        import pyarrow as pa
        d = sys['work_dedup']['description'].fillna('')
        has = np.flatnonzero((d.str.len() > 20).to_numpy())
        sys['desc_ids'] = has
        sys['desc_text'] = pa.array([fold(x) for x in d.iloc[has]], type=pa.string())
    return sys['desc_ids'], sys['desc_text']


def mention_ids(sys, phrase):
    """Books whose description names the phrase ("percy jackson", "narnia", "hermione"): a series or a
    character that is in no title. Most-read first.
    Speed: a native substring search for the longest word first, then the exact phrase only on those."""
    import pyarrow.compute as pc
    words = re.findall(r'[a-z0-9]+', fold(phrase))
    if not words or not 1 <= len(words) <= 3 or sum(len(w) for w in words) < 4:
        return EMPTY_IDS
    ids, text = _mention_index(sys)
    first = max(words, key=len)
    rough = np.flatnonzero(pc.match_substring(text, first).to_numpy(zero_copy_only=False))
    if not len(rough):
        return EMPTY_IDS
    pat = re.compile(r'(?<![a-z0-9])' + r'[\s_]+'.join(re.escape(w) for w in words) + r'(?![a-z0-9])')
    sub = text.take(rough).to_pylist()
    hit = ids[rough[[bool(pat.search(t)) for t in sub]]]
    if not len(hit):
        return EMPTY_IDS
    rc = sys['work_dedup']['rating_count'].fillna(0).to_numpy()[hit]
    return hit[np.argsort(-rc, kind='stable')]


def _top_pop(sys, ids):
    """How much the 10 most-read books among `ids` are read (same measure as spelling_alternatives)."""
    if not len(ids):
        return 0.0
    p = sys['work_dedup']['rating_count'].fillna(0).to_numpy()[ids]
    return float(np.partition(p, len(p) - 10)[-10:].sum()) if len(p) > 10 else float(p.sum())


def popular_neighbours(sys, query, keep=2):
    """A correctly spelled word can still be a slip for a far more read one: "hobbs" (Will Hobbs) is
    1-2 letters from hobbit and hobby, "hawkings" from Hawking. Returns up to `keep` queries with one
    word swapped for such a neighbour, most read first. The typed query itself is NOT replaced: these
    are only offered as "Did you mean" links and small sections under the real results."""
    tokens = query.split()
    if not 1 <= len(tokens) <= 4:
        return []
    words = [fold(t).strip('.,!?;:"()') for t in tokens]

    def together(ws):
        """Books that contain ALL the (known) words: a swapped word is judged with the rest of the query,
        so "sherlock holmes" is never offered "sherlock hommes" (no book has both of those)."""
        ids = None
        for w in ws:
            if not (w.isalpha() and len(w) >= 3) or w in STOPWORDS:
                continue
            d = docs_with(sys, w)
            ids = d if ids is None else np.intersect1d(ids, d)
        return EMPTY_IDS if ids is None else ids

    found = []
    for i, tok in enumerate(tokens):
        b = words[i]
        if not (b.isalpha() and len(b) >= 4 and b not in STOPWORDS):
            continue
        own_ids = docs_with(sys, b)
        if not len(own_ids):
            continue                                       # an unknown word is spelling_alternatives' job
        own_pop = _top_pop(sys, together(words))
        limit = 1 if len(b) <= 4 else 2
        for cand, _, _ in process.extract(b, sys['fix_words'], scorer=fuzz.ratio, score_cutoff=70, limit=40):
            if cand == b or cand[0] != b[0] or cand in STOPWORDS:
                continue
            edits = DamerauLevenshtein.distance(b, cand)
            if edits > limit or (edits == 2 and cand[:3] != b[:3]):   # hob-bs ~ hob-bit; fro-do is not fyo-dor
                continue
            pop = _top_pop(sys, together([cand if j == i else w for j, w in enumerate(words)]))
            if pop >= 3 * max(own_pop, 1) and pop >= 100:     # only a link, so a lower bar than replacing
                swapped = ' '.join(cand if j == i else t for j, t in enumerate(tokens))
                found.append((pop / (1 + edits), swapped))
    found.sort(reverse=True)
    out = []
    for _, q in found:
        if q not in out:
            out.append(q)
        if len(out) == keep:
            break
    return out


PHRASE_SKIP = {'the', 'a', 'an', 'of', 'and', 'to', 'in', 'on', 'for', 'with', 'by', 'at', 'from'}


def popular_phrase_index(sys):
    """Two- and three-word phrases from titles and author names, with how much the books that hold them are read
    ("harry potter" is read far more than "harry power"). Built once (under a second), then kept."""
    idx = sys.get('phrase_index')
    if idx is not None:
        return idx
    wd = sys['work_dedup']
    rc = pd.to_numeric(wd['rating_count'], errors='coerce').fillna(0).to_numpy()
    titles = wd['title'].fillna('').to_numpy()
    pops = {}
    for i in np.flatnonzero(rc >= 3):
        words = proc(re.split(r'[:(\[]', titles[i])[0]).split()[:7]
        seen = set()
        for n in (2, 3, 4):
            for j in range(len(words) - n + 1):
                g = words[j:j + n]
                if g[-1] in PHRASE_SKIP or (n == 2 and g[0] in PHRASE_SKIP):
                    continue                              # not "of the" or "game of"; "to kill a mockingbird" is fine
                p = ' '.join(g)
                if p not in seen:
                    seen.add(p)
                    pops[p] = pops.get(p, 0.0) + float(rc[i])
    for a, fame in sys['author_fame'].items():                  # "stephen king" counts as a phrase too
        if fame >= 20:
            name = proc(sys['author_display'].get(a, a))
            if 2 <= len(name.split()) <= 3:
                pops[name] = pops.get(name, 0.0) + float(fame)
    pops = {p: v for p, v in pops.items() if v >= 20}
    idx = {'pop': pops}
    for n in (2, 3, 4):
        names = [p for p in pops if len(p.split()) == n]
        idx[n] = (names, [pops[p] for p in names])
    sys['phrase_index'] = idx
    return idx


def phrase_suggestions(sys, query, keep=2):
    """"harry power" is two real words, yet one step from "harry potter", read far more: like a search engine,
    offer it as "Did you mean". Only for 2-3 word queries that are not themselves a well-read phrase."""
    words = proc(query or '').split()
    if len(words) > 4:
        return []
    if len(words) not in (2, 3, 4):
        return []
    idx = popular_phrase_index(sys)
    typed = ' '.join(words)
    typed_pop = idx['pop'].get(typed, 0.0)
    if typed_pop >= 60:                                      # what was typed is already a well-read phrase
        return []
    names, pops = idx[len(words)]
    found = []
    # a word our books have never seen makes a looser match fair ("hunegr ganed" -> hunger games)
    unknown = any(w.isalpha() and len(w) >= 3 and w not in PHRASE_SKIP and w not in sys['vocab'] for w in words)
    for ph, score, i in process.extract(typed, names, scorer=fuzz.ratio, score_cutoff=72 if unknown else 80, limit=20):
        if ph == typed or pops[i] < 150 or pops[i] < 10 * max(typed_pop, 5):
            continue
        if any(DamerauLevenshtein.distance(a, b) > 3 for a, b in zip(typed.split(), ph.split())):
            continue                                         # no word changed by more than 3 letters
        found.append((score + 8 * math.log10(pops[i]), ph))
    found.sort(reverse=True)
    return [ph for _, ph in found[:keep]]


def phrase_completion(sys, query):
    """A half-typed phrase, like a search box that finishes your words: "harry p" -> harry potter, "george o" ->
    george orwell, "game of t" -> game of thrones, "machine l" -> machine learning. The most-read phrase that
    starts with the typed words wins. Only when the last word is 1-4 letters and the typed text is no phrase itself."""
    words = re.findall(r"[a-z0-9]+", fold(query or ''))      # not proc(): it would merge "a m" into "am"
    if len(words) not in (2, 3, 4) or not words[-1].isalpha() or not 1 <= len(words[-1]) <= 4:
        return None
    idx = popular_phrase_index(sys)
    typed = ' '.join(words)
    if idx['pop'].get(typed, 0.0) >= 60:
        return None
    head, last = ' '.join(words[:-1]) + ' ', words[-1]
    best = None
    for n in range(len(words), 5):                           # the phrase may be longer: "lord of t" -> lord of the rings
        names, pops = idx[n]
        for ph, p in zip(names, pops):
            if p >= 20 and ph.startswith(head) and ph[len(head):].startswith(last) and len(ph) > len(typed):
                if best is None or p > best[1]:
                    best = (ph, p)
    return best[0] if best else title_completion(sys, words)


def title_completion(sys, words):
    """No well-read phrase starts with the typed words (textbooks have few ratings): read the next word straight
    from the titles that contain the earlier words ("machine l" -> machine learning)."""
    head, last = words[:-1], words[-1]
    ids = None
    for w in head:
        if w in PHRASE_SKIP or len(w) < 3:
            continue
        d = docs_with(sys, w)
        ids = d if ids is None else np.intersect1d(ids, d)
    if ids is None or not len(ids):
        return None
    wd = sys['work_dedup']
    titles = wd['title'].to_numpy()
    rc = pd.to_numeric(wd['rating_count'], errors='coerce').fillna(0).to_numpy()
    prefix = ' '.join(head) + ' '
    counts = {}
    for i in ids[:6000]:
        t = proc(re.split(r'[:(\[]', str(titles[i]))[0])
        k = t.find(prefix)
        while k != -1:
            rest = t[k + len(prefix):].split()
            if rest and rest[0].startswith(last) and rest[0] != last and rest[0] not in PHRASE_SKIP:
                counts[rest[0]] = counts.get(rest[0], 0.0) + 1 + float(rc[i])
            k = t.find(prefix, k + 1)
    if not counts:
        return None
    best = max(counts, key=counts.get)
    return prefix + best if counts[best] >= 3 else None


def _phrase_sound(s):
    """'The Lord of the Rings' -> 'lord of th rings': how a whole phrase sounds, without a leading article
    (so "lrd of th rngs" is not pushed toward shorter titles that lack the "The")."""
    words = proc(s).split()
    if len(words) > 1 and words[0] in ('the', 'a', 'an'):
        words = words[1:]
    return ' '.join(sounds_like(w) for w in words)


def _sound_index(sys, name):
    """Lazily built 'how it sounds' keys for every title / author name (made once, then reused)."""
    key = name + '_sound'
    if key not in sys:
        src = sys['title_list'] if name == 'title' else sys['author_list']
        sys[key] = [_phrase_sound(s) for s in src]
    return sys[key]


def phrase_guesses(sys, query, keep=3):
    """Very misspelled queries (4+ wrong letters: "shrlck hlms", "hary pottr and filosofers ston"):
    no single word is close to anything, but the WHOLE phrase is close to a real title or author,
    especially when compared by sound. Returns up to `keep` titles / author names, most likely first."""
    q_sound = _phrase_sound(query)
    if len(q_sound.replace(' ', '')) < 5:
        return []
    pop = sys.get('title_pop')
    if pop is None:                               # built once, then reused (it covers all 500,000 books)
        pop = sys['title_pop'] = sys['work_dedup'].drop_duplicates('title').set_index('title')['rating_count'].to_dict()
    fame = sys['author_fame']
    scored = []
    for kind in ('title', 'author'):
        names = sys['title_list'] if kind == 'title' else sys['author_list']
        for _, score, i in process.extract(q_sound, _sound_index(sys, kind), scorer=fuzz.ratio,
                                           score_cutoff=72, limit=8):
            name = names[i]
            reads = pop.get(name, 0) if kind == 'title' else fame.get(sys['author_norm_of'].get(name, ''), 0)
            scored.append((score + 6 * math.log10(1 + (reads or 0)), name))
    scored.sort(reverse=True)
    out = []
    for _, name in scored:
        if name.lower() not in (o.lower() for o in out) and fold(name) != fold(query):
            out.append(name)
        if len(out) == keep:
            break
    return out


def sharing_ids(sys, kind, pq):
    """Titles / author names that have a word starting with the same 2 letters as a typed word, or None
    when nothing narrows it down (then everything is compared). A close match always shares a word start."""
    # first 2 letters: a typo from the 3rd letter on still matches ("rolling" -> Rowling)
    keys = {w[:2] for w in pq.split() if len(w) >= 3 and w not in STOPWORDS}
    if not keys:
        return None
    ids_of, words = sys[kind + '_word_ids'], sys[kind + '_words']
    found = []
    for k in keys:
        lo, hi = bisect.bisect_left(words, k), bisect.bisect_left(words, k + '￿')
        found.extend(ids_of[w] for w in words[lo:hi])
    if not found:
        return None
    ids = np.unique(np.concatenate(found))
    total = len(sys['title_proc'] if kind == 'title' else sys['author_proc'])
    return ids if len(ids) < 0.5 * total else None


def closest_title(sys, q, cutoff=0):
    """Closest catalog title -> (title_proc, score, index) or None. Compares in word order AND with the words
    sorted, and keeps the better: sorting alone breaks on a typo that changes the order
    ("gone with the wnd": sorted "...with wnd" vs "...wind with" scores 74; in order it is 97).
    Speed: only titles that have a word starting with the same 3 letters as a typed word are compared
    (a close title always shares words with the query); all titles only when there is no such title."""
    pq = proc(q)
    titles = sys['title_proc']
    cand = sharing_ids(sys, 'title', pq)
    pool = titles if cand is None else [titles[i] for i in cand]
    a = process.extractOne(pq, pool, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff)
    b = process.extractOne(pq, pool, scorer=fuzz.ratio, score_cutoff=cutoff)
    found = [x for x in (a, b) if x]
    if not found:
        return None
    best = max(found, key=lambda x: x[1])
    return best if cand is None else (best[0], best[1], int(cand[best[2]]))


def understand_query(original_query, sys, _fixed=False):
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

    tb = closest_title(sys, q)
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
            if pick.get('others'):
                out['similar'] = True
            if corrected:
                tw = one if one in sys['vocab'] else best_title_word(sys, one)
                if tw:
                    out['title_word'] = tw
                    if tw != one:
                        out['title_from_typo'] = tw.title()
            return {**out, 'kind': 'author', 'author_norm': norm, 'author_name': sys['author_display'][norm],
                    'corrected': corrected, 'said': pick['name'],
                    'also_title': title_best if corrected and title_score >= 90 else None,   # only a very close title
                    'also_authors': list(dict.fromkeys(
                        [sys['authors_by_name'][o][0] for o in pick.get('others', [])
                         if sys['authors_by_name'][o][0] != norm] + other_authors(sys, authors[1:], norm)))[:5]}

    # 3. a full author name ("stephen king", "jk roling") or one not-everyday word that is an author
    toks = proc(q).split()
    if n >= 2 and len(toks) >= 2 and toks[-1] in sys['authors_by_name'] and all(t.isalpha() and len(t) <= 3 for t in toks[:-1]):
        # initials + surname: "d e knuth" -> Donald E. Knuth, "th cormen" -> Thomas H. Cormen
        typed_init = ''.join(toks[:-1])
        for norm in sys['authors_by_name'][toks[-1]]:              # best-known first
            parts = proc(sys['author_display'].get(norm, norm)).split()
            if len(parts) < 2 or parts[-1] != toks[-1]:
                continue
            given = parts[:-1]
            firsts = ''.join(p[0] for p in given)
            mixed = ''.join(p if len(p) <= 3 else p[0] for p in given)
            if typed_init in (firsts, mixed) or (len(typed_init) >= 2 and firsts.startswith(typed_init)):
                return {**out, 'kind': 'author', 'author_norm': norm, 'author_name': sys['author_display'][norm],
                        'corrected': False, 'said': q, 'also_title': None, 'also_authors': []}

    if looks_like_name and not everyday_word:
        a_cand = sharing_ids(sys, 'author', proc(q))           # only names sharing a word start (much faster)
        a_pool = sys['author_proc'] if a_cand is None else [sys['author_proc'][i] for i in a_cand]
        a_map = (lambda i: i) if a_cand is None else (lambda i: int(a_cand[i]))
        matches = [(m, sc, a_map(i)) for m, sc, i in
                   process.extract(proc(q), a_pool, scorer=fuzz.token_sort_ratio, limit=8)]
        strong = [(sys['author_list'][i], s) for _, s, i in matches if s >= 85]
        # names that contain every typed word ("garcia marquez" -> Gabriel Garcia Marquez,
        # "ronald tolkien" -> John Ronald Reuel Tolkien), when the typed last word is a known surname
        full = []
        typed_last = proc(q).split()[-1] if proc(q).split() else ''
        if n >= 2 and typed_last in sys['authors_by_name'] and title_score < 85:   # not "harry potter", "artificial intelligence"
            typed_set = set(proc(q).split())
            full = [sys['author_list'][a_map(i)] for _, _, i in process.extract(
                proc(q), a_pool, scorer=fuzz.token_set_ratio, score_cutoff=100, limit=8)
                if typed_set <= set(sys['author_proc'][a_map(i)].split())]   # the name has EVERY typed word ("Jackson" is not "percy jackson")
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
            corrected = False
            fame = sys['author_fame']
            famous_near = [sys['author_norm_of'][a] for a, sc in strong if sc >= 88]
            if famous_near:                       # "jane austin" -> Jane Austen, not a little-known Jane Austin
                star = max(famous_near, key=lambda x: fame.get(x, 0))
                close = DamerauLevenshtein.distance(proc(q), proc(sys['author_display'].get(star, star))) <= 2
                if close and fame.get(star, 0) >= 50 and fame.get(star, 0) >= 10 * max(fame.get(norm, 0), 1):
                    norm, corrected = star, True
            also = []
            if n == 1 and one in sys['authors_by_name']:     # "arthur": the other well-known Arthurs too
                also = other_authors(sys, sys['authors_by_name'][one][1:], norm)
            return {**out, 'kind': 'author', 'author_norm': norm, 'author_name': sys['author_display'][norm],
                    'corrected': corrected, 'said': sys['author_display'][norm] if corrected else (one or q),
                    'also_title': None, 'also_authors': also}

    # 3b. misspelled words: fix them with the catalog's own words, then read the fixed query from the top
    #     ("jk rolimg" -> jk rowling = an author; "lord of the rigns" -> a title; "chilfrend's book" -> words)
    if not _fixed and has_unknown_word:
        fixed = fix_typos(sys, q)
        if fixed != q:
            r = understand_query(fixed, sys, _fixed=True)
            if r['kind'] == 'author':
                r['corrected'], r['said'] = True, fixed
            return r

    # 4. a misspelled title: only when a word looks like a typo, so real words stay as typed
    #    ("introduction to physics" must not become "...Metaphysics")
    if 85 <= title_score < 100 and n <= 4 and (has_unknown_word or (has_rare_word and title_score >= 90)):
        return {**out, 'kind': 'title_fix', 'query': title_best}

    # 4b. words typed without spaces ("lordoftherings"): only when no title is close
    #     (a misspelling like "frankenstine" was already fixed by step 4, not split into "frank en stine")
    if n == 1 and len(one) >= 7 and one.isalpha() and one not in sys['vocab'] and not len(docs_with(sys, one)):
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
            pool = in_category(pool, category_filter)
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
            df = in_category(df, category_filter)
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

        who = f"Showing results for author: {name}"      # a corrected name ("stfen kng") just shows the real one
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
            more['section'] = (f"Other authors named {said}" if not (reading['corrected'] or reading.get('similar'))
                               else "Authors with a similar name")
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
    rows_by_isbn = sys['rows_by_isbn']
    sem = np.zeros(len(work_dedup))
    for isbn, cos in near_cos.items():
        rows = rows_by_isbn.get(isbn)
        if rows is not None:
            sem[rows] = cos
    semantic_cos = pd.Series(sem, index=work_dedup.index)

    faiss_score_by_isbn = {}
    for idx, dist in zip(faiss_indices[0], distances[0]):
        isbn = isbn13_by_position[idx]
        if not isinstance(isbn, str):
            continue            # NaN ISBN: as a dict key it would give this score to all ~5,800 books without an ISBN
        sim = 1 - (dist / max_dist)
        if isbn not in faiss_score_by_isbn or sim > faiss_score_by_isbn[isbn]:
            faiss_score_by_isbn[isbn] = sim

    faiss_scores = np.zeros(len(work_dedup))
    for isbn, sim in faiss_score_by_isbn.items():
        rows = rows_by_isbn.get(isbn)
        if rows is not None:
            faiss_scores[rows] = sim

    max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1
    bm25_norm = bm25_scores / max_bm25

    pop_norm = sys['pop_norm']

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
            ids = docs_with_forms(sys, parts[0])
            for x in parts[1:]:
                ids = np.intersect1d(ids, docs_with_forms(sys, x), assume_unique=True)
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
            pat = r'(?<!\w)' + re.escape(w) if w == partial_word else form_pattern(w)   # unfinished: "starts with"
            ok &= col.str.contains(pat, regex=True).to_numpy()
        hit[candidates[ok]] = True
        return pd.Series(hit, index=work_dedup.index)

    title_hit = words_in('title')
    author_hit = words_in('author')
    primary_hit = author_hit if wants_author else title_hit

    # a book is RELATED to the query if it shares a search word, or is close in meaning.
    # Author cue words ("by", "written") don't count: "books by dickens" must not match "Squeak by Example".
    # With 2+ search words, sharing just one common word isn't enough ("history of rome" must not match every
    # Computer Science book that mentions "history"): at least two of the words must appear.
    joining = {'of', 'in', 'on', 'and', 'to', 'for', 'with', 'by', 'from', 'at', 'or', 'as', 'into', 'its', 'is'}
    meaningful = [w for w in dict.fromkeys(q_words) if w not in joining] or q_words   # "of" is not a match
    if len(meaningful) >= 2:
        words_hit = np.zeros(len(work_dedup), dtype=np.int16)
        for w in meaningful:
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

    combined_scores = np.asarray((bm25_weight * bm25_norm) + (faiss_weight * faiss_scores) + (pop_weight * pop_norm)
                                 + (0.40 * primary_hit.values) + (0.10 * sys['has_read']), dtype=np.float64)
    common_words = {'a', 'the', 'book', 'about', 'story', 'something', 'like', 'that'}
    query_words = set(query.lower().split())
    looks_like_name_query = len(query.split()) <= 4 and len(query) >= 4 and not (query_words & common_words)

    # Only books that can appear on the page are copied and sorted: the best-scoring few thousand, every exact
    # title/author match (they are listed by popularity in their sections), the books close in meaning, and -
    # with a filter on - every related book. Same results as sorting all 500,000 books, much faster.
    keep_n = min(4000, len(combined_scores))
    parts = [np.argpartition(-combined_scores, keep_n - 1)[:keep_n] if keep_n else EMPTY_IDS.astype(np.int64),
             np.flatnonzero(title_hit.to_numpy()), np.flatnonzero(author_hit.to_numpy()),
             np.flatnonzero(sem >= RELATED_COS)]
    if filters_on:
        parts.append(np.flatnonzero(related.to_numpy()))
    pool = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in parts]))
    results = work_dedup.iloc[pool].copy()
    results['score'] = combined_scores[pool]

    suggestion = None
    if combined_scores.max() < 0.15:
        corrections = suggest_correction(query, work_dedup)
        if corrections:
            suggestion_text, _ = corrections[0]
            suggestion = suggestion_text

    if category_filter and category_filter in CATEGORY_GROUPS:
        results = in_category(results, category_filter)

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

    # stable sort: books with exactly equal scores keep catalog order, so the page is the same every time
    ranked = results.sort_values('score', ascending=False, kind='stable').drop_duplicates('dedup_key')
    ranked = ranked.assign(rank_pop=ranked['n_ratings'] + 50 * (ranked['read_url'] != ''))
    if sectioned:
        t_all = ranked[ranked['section'] == TITLE_LABEL].sort_values(['n_ratings', 'rating_count'], ascending=False)
        # A study subject ("calculus", "organic chemistry"): most title matches are textbooks. Then real
        # textbooks go first, free ones (readable here) at the top, and novels/comics that merely share the
        # word (Tintin's "Calculus Affair") go last.
        if len(t_all) >= 6:
            cats = t_all['categories'].fillna('')
            acad = cats.str.contains('|'.join(p for g in ACADEMIC_GROUPS for p in CATEGORY_GROUPS[g]), case=False, regex=True)
            acad &= ~cats.str.contains(NOT_ACADEMIC, case=False, regex=True)
            fiction = cats.str.contains(NOT_ACADEMIC, case=False, regex=True)
            study_query = bool(re.search(STUDY_WORDS, ' '.join(q_words)))
            if study_query or acad.head(20).mean() >= 0.5:
                free = (t_all['read_url'] != '')
                key = (~fiction).astype(int) * 2 + (~fiction & free).astype(int)
                t_all = t_all.assign(_k=key.to_numpy()).sort_values('_k', ascending=False, kind='stable').drop(columns='_k')
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
    # does ANY book have every typed word in its title + author together? ("percy jackson": books by
    # Percy Shelley share "percy" and feel close in meaning, but no book has both words -> not here)
    together = False
    if q_words and candidates is not None and len(candidates):
        both = work_dedup['title_f'].iloc[candidates] + ' ' + work_dedup['author_f'].iloc[candidates]
        ok = np.ones(len(candidates), dtype=bool)
        for w in q_words:
            ok &= both.str.contains(form_pattern(w), regex=True).to_numpy()
        together = bool(ok.any())
    if (not filters_on and not suggestion and len(q_words) >= 2 and not wants_author and looks_like_name_query
            and not title_hit.any() and not author_hit.any() and (best_cos < 0.60 or not together)):
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
        pool = in_category(pool, category_filter)
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
    'db': 'database',
}

# Short forms that also mean something else ("hp" is H.P. Lovecraft, "got" is a plain word): what was typed is
# searched as typed, and the famous meaning is offered as a "Did you mean" link.
AMBIGUOUS_SHORT = {'hp': ['harry potter'], 'got': ['game of thrones'], 'asoiaf': ['a song of ice and fire'],
                   'twd': ['the walking dead'], 'lotf': ['lord of the flies'], 'tkam': ['to kill a mockingbird']}


# Real name -> the pen name the books are filed under (and a few pen names -> the famous real name).
# Typed with small typos too: matched with a close spelling.
PEN_NAMES = {
    'eric blair': 'george orwell', 'eric arthur blair': 'george orwell',
    'samuel clemens': 'mark twain', 'samuel langhorne clemens': 'mark twain',
    'charles dodgson': 'lewis carroll', 'charles lutwidge dodgson': 'lewis carroll',
    'theodor geisel': 'dr. seuss', 'theodor seuss geisel': 'dr. seuss', 'ted geisel': 'dr. seuss',
    'mary ann evans': 'george eliot', 'marian evans': 'george eliot',
    'francois marie arouet': 'voltaire', 'daniel handler': 'lemony snicket',
    'amandine dupin': 'george sand', 'aurore dupin': 'george sand',
    'hector hugh munro': 'saki', 'william sydney porter': 'o. henry',
    'currer bell': 'charlotte bronte', 'ellis bell': 'emily bronte', 'acton bell': 'anne bronte',
    'mary westmacott': 'agatha christie',
    'joanne rowling': 'j.k. rowling', 'joanne kathleen rowling': 'j.k. rowling',
}


def pen_name_of(query):
    """'eric blair' (or 'eric blare') -> ('george orwell', 'Eric Blair'), else None."""
    q = ' '.join(re.findall(r'[a-z]+', fold(query)))
    if len(q) < 6:
        return None
    hit = process.extractOne(q, list(PEN_NAMES), scorer=fuzz.ratio, score_cutoff=92)
    if hit and PEN_NAMES[hit[0]] != hit[0]:
        return PEN_NAMES[hit[0]], hit[0].title()
    return None


ISBN_WORD = re.compile(r'^\s*isbn\s*[:#-]?\s*', re.I)
ISBN_LABEL = re.compile(r'^\s*isbn[\s-]*1[03]\s*[:#-]?\s*', re.I)     # "ISBN-13: ..."


def isbn_check_ok(digits):
    """Is the last digit right? (A mistyped digit almost always breaks it.)"""
    d = digits.upper()
    if len(d) == 10:
        total = sum((10 - i) * (10 if c == 'X' else int(c)) for i, c in enumerate(d))
        return total % 11 == 0
    if len(d) == 13 and d.isdigit():
        return sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(d)) % 10 == 0
    return False


# Short words students use inside a query; each is replaced by the word book titles use.
WORD_FORMS = {'maths': 'mathematics', 'math': 'mathematics', 'intro': 'introduction', 'stats': 'statistics',
              'econ': 'economics', 'algo': 'algorithms', 'algos': 'algorithms', 'prog': 'programming',
              'engg': 'engineering', 'bio': 'biology', 'chem': 'chemistry', 'phys': 'physics',
              'psych': 'psychology', 'programing': 'programming'}

def isbn_search(isbn, sys):
    """A typed ISBN (10 or 13 digits) -> that book, whichever edition the number belongs to."""
    isbn13 = to_isbn13(isbn.upper()) or isbn
    wd, books = sys['work_dedup'], sys['books']
    keys = set(books.loc[books['isbn13'] == isbn13, 'dedup_key'])
    found = wd[wd['dedup_key'].isin(keys) | (wd['isbn13'] == isbn13)].drop_duplicates('dedup_key')
    if len(found):
        return found.head(24), f"Book with ISBN {isbn}"
    if not isbn_check_ok(isbn):
        return found, f"“{isbn}” is not a valid ISBN: a digit is probably mistyped. Check the number on the book."
    return found, f"No book with ISBN {isbn} in our library."


def _add_mentions(sys, df, phrase, banner, category_filter=None, free_only=False):
    """A series or character that is in no title ("percy jackson", "narnia"): add the books whose description
    names it. When they are read far more than what the normal search found, they go FIRST."""
    words = [w for w in re.findall(r'[a-z]+', fold(phrase or '')) if w not in STOPWORDS]
    if not 1 <= len(words) <= 3 or len(''.join(words)) < 4:
        return df, banner
    ids = mention_ids(sys, ' '.join(words))
    if not len(ids):
        return df, banner
    m = sys['work_dedup'].iloc[ids[:200]]
    if category_filter in CATEGORY_GROUPS:
        m = in_category(m, category_filter)
    if free_only:
        m = m[m['read_url'] != '']
    if 'dedup_key' in df.columns:
        m = m[~m['dedup_key'].isin(set(df['dedup_key']))]
    m = m.drop_duplicates('dedup_key').head(8).copy()
    if not len(m):
        return df, banner
    reads = lambda d: float(pd.to_numeric(d['rating_count'], errors='coerce').fillna(0).head(5).sum())         if 'rating_count' in d.columns and len(d) else 0.0
    m_pop, main_pop = reads(m), reads(df)
    phrase_hit = len(words) >= 2                              # a whole name like "percy jackson" is very specific
    if m_pop < 100 or (not phrase_hit and m_pop < 0.5 * main_pop):
        return df, banner                                    # nothing well known: keep the page as it is
    nice = ' '.join(w.title() for w in words)
    m['section'] = f"Books about “{nice}”"
    if len(df) and ('section' not in df.columns or not df['section'].fillna('').astype(bool).any()):
        df = df.copy()
        df['section'] = f"Results for “{phrase.strip()}”"
    # an everyday topic word ("loneliness", "dragons") with books TITLED by it: those stay first, the books
    # whose description mentions it come after. Names ("percy", "narnia") may still put the series first.
    topic = not phrase_hit and all(spell.word_usage_frequency(w) >= 1e-6 for w in words)
    titled = topic and len(df) and 'title' in df.columns and df['title'].fillna('').map(fold).str.contains(
        r'(?<!\w)' + re.escape(words[0])).head(10).any()
    if not titled and m_pop >= (1 if phrase_hit else 3) * max(main_pop, 1):
        df = pd.concat([m, df], ignore_index=True)
        if (banner or '').startswith(('“', 'No books match')):   # "isn't in our library": it is, under another title
            banner = f"Showing books about “{nice}”."
    else:
        df = pd.concat([df, m.head(6)], ignore_index=True)
    return df, banner


# Famous characters -> the series (as a search that finds it). Book data rarely names characters, so a
# student typing "hermione" or "katniss" would otherwise only get unrelated books.
CHARACTERS = {
    'Harry Potter': ['hermione', 'hermione granger', 'ron weasley', 'dumbledore', 'albus dumbledore', 'voldemort',
                     'snape', 'severus snape', 'hagrid', 'draco malfoy', 'hogwarts', 'sirius black', 'neville longbottom'],
    'Lord of the Rings': ['frodo', 'frodo baggins', 'gandalf', 'aragorn', 'legolas', 'gimli', 'samwise', 'gollum',
                          'sauron', 'middle earth', 'middle-earth'],
    'The Hobbit': ['bilbo', 'bilbo baggins', 'smaug', 'thorin'],
    'Hunger Games': ['katniss', 'katniss everdeen', 'peeta', 'peeta mellark', 'district 12', 'mockingjay'],
    'Sherlock Holmes': ['sherlock', 'dr watson', 'moriarty', 'baker street'],
    'Narnia': ['aslan', 'lucy pevensie', 'pevensie', 'narnia'],
    'Percy Jackson': ['annabeth', 'camp half-blood', 'camp half blood'],
    'Twilight': ['bella swan', 'edward cullen', 'jacob black'],
    'Game of Thrones': ['jon snow', 'daenerys', 'tyrion', 'tyrion lannister', 'westeros', 'arya stark', 'cersei'],
    'Hercule Poirot': ['poirot', 'hercule poirot', 'miss marple'],
    'Pride and Prejudice': ['elizabeth bennet', 'mr darcy', 'darcy'],
    'Alice in Wonderland': ['mad hatter', 'cheshire cat'],
    'Winnie the Pooh': ['pooh', 'piglet', 'eeyore', 'christopher robin'],
    'Dune': ['paul atreides', 'arrakis', 'muad dib'],
    'Diary of a Wimpy Kid': ['greg heffley'],
    'Nancy Drew': ['nancy drew'],
    'Hardy Boys': ['hardy boys'],
    'Artemis Fowl': ['artemis fowl'],
    'Tintin': ['tintin', 'captain haddock', 'snowy'],
    'Discworld': ['rincewind', 'ankh morpork', 'granny weatherwax'],
}
_CHARACTER_OF = {name: series for series, names in CHARACTERS.items() for name in names}


def character_series(query):
    """'hermione' / 'hermoine' -> ('Hermione', 'Harry Potter'), else None. Small typos allowed (5+ letters)."""
    q = ' '.join(re.findall(r'[a-z0-9-]+', fold(query or '')))
    if not q:
        return None
    if q in _CHARACTER_OF:
        return q.title(), _CHARACTER_OF[q]
    if len(q) >= 5:
        hit = process.extractOne(q, list(_CHARACTER_OF), scorer=fuzz.ratio, score_cutoff=86)
        if hit:
            return hit[0].title(), _CHARACTER_OF[hit[0]]
    return None


BY_TOPIC = re.compile(r'(?:^|\s)by\s+(?P<author>[^,]+?)(?:\s+(?:about|on|with|featuring|where|that)\s+(?P<topic>.+))?$')


def author_topic(query, sys, category_filter=None, free_only=False, top_k=24):
    """'books by stephan king about clowns' -> that author's books, the ones closest to the topic first.
    Returns (df, banner) or None when the sentence has no 'by <author> about <topic>' shape."""
    m = BY_TOPIC.search(fold(query or '').strip())
    if not m:
        return None
    tb = closest_title(sys, query, cutoff=92)
    if tb:
        return None                               # a real title with "by" in it ("stand by me", "death by chocolate")
    reading = understand_query(m['author'], sys)
    if reading.get('kind') != 'author':
        return None
    wd = sys['work_dedup']
    books = wd[wd['author_norm'] == reading['author_norm']]
    if category_filter in CATEGORY_GROUPS:
        books = in_category(books, category_filter)
    if free_only:
        books = books[books['read_url'] != '']
    books = books.drop_duplicates('dedup_key')
    if not len(books):
        return None
    if not m['topic']:
        # "books by stephen king": that author's books, most-read first (not books ABOUT Stephen King)
        df = books.sort_values(['n_ratings', 'rating_count'], ascending=False).head(top_k).copy()
        df['section'] = f"Books by {reading['author_name']}"
        return df, f"Showing results for author: {reading['author_name']}"
    topic = m['topic'].strip()
    # closeness in meaning between the topic and each of the author's books (book vectors are unit length)
    if 'emb_row_of_isbn' not in sys:
        sys['emb_row_of_isbn'] = {isbn: i for i, isbn in enumerate(sys['isbn13_by_position']) if isinstance(isbn, str)}
    qv = sys['model_embed'].encode([topic]).astype('float32')[0]
    rows = [sys['emb_row_of_isbn'].get(i) for i in books['isbn13']]
    emb = sys['embeddings']
    sim = np.array([float(emb[r] @ qv) if r is not None else 0.0 for r in rows])
    words = [w for w in re.findall(r'[a-z]+', fold(topic)) if w not in STOPWORDS and len(w) > 2]
    text = (books['title'].fillna('') + ' ' + books['description'].fillna('')).map(fold)
    word_hit = np.array([sum(1 for w in words if w[:5] in t) for t in text], dtype=float)
    pop = np.log1p(pd.to_numeric(books['rating_count'], errors='coerce').fillna(0).to_numpy())
    order = np.argsort(-(sim + 0.3 * word_hit + 0.02 * pop), kind='stable')
    df = books.iloc[order].head(top_k).copy()
    df['section'] = f"Books by {reading['author_name']}"
    return df, f"Books by {reading['author_name']} about “{topic}”, closest match first."


def join_apostrophe_names(query, sys):
    """"o'brien" -> "obrien", "d'angelo" -> "dangelo": the catalog files these surnames without the
    apostrophe, and people type them with it. Only joined when an author really has the joined surname."""
    def fix(m):
        joined = fold(m.group(1) + m.group(2))
        return joined if joined in sys['authors_by_name'] else m.group(0)
    return re.sub(r"\b([od])['’]([a-zA-Z]{3,})\b", fix, query or '', flags=re.I)


def exact_title_first(df, query):
    """A title typed in full ("scala book") goes first when nothing at the top even contains all its words
    (otherwise "La Scala" novels, read more, push the exact book down)."""
    words = proc(query).split()
    if len(words) < 2 or len(df) < 2 or 'title' not in df.columns:
        return df
    clean = df['title'].fillna('').map(lambda t: proc(re.split(r'[:(\[]', t)[0]))
    exact = (clean == ' '.join(words)).to_numpy()
    if not exact.any() or exact[:3].any():
        return df
    top_has_all = [all(w in proc(t).split() for w in words) for t in df['title'].fillna('').head(5)]
    if any(top_has_all):
        return df
    out = pd.concat([df[exact], df[~exact]])
    out.attrs.update(df.attrs)
    return out


LOOKALIKES = {'@': 'a', '4': 'a', '0': 'o', '3': 'e', '$': 's', '5': 's', '7': 't', '1': ['i', 'l']}
INITIALS = re.compile(r'(?:[^\W\d_]\.)+')


def tidy_query(query, sys):
    """Symbols are noise, as they are for a search engine: "harry potter!!", "harry.potter", "harry,potter",
    "$potter", "#harrypotter", '"harry potter"' and "harry potter&the chamber" all mean the plain words.
    A letter swapped for a look-alike symbol or digit ("h@rry", "pott3r", "k1ng") is read as the letter, but only
    when that gives a word our books have, so "catch 22", "1984" and "fahrenheit 451" stay as they are."""
    q = query or ''
    if len(re.sub(r'\D', '', q)) >= 8:                       # an ISBN: leave it alone
        return q
    if re.search(r'\w&\w|\w&\s|\s&\w', q):
        q = re.sub(r'\s*&\s*', ' & ', q)                    # "potter&the" -> "potter & the"
    q = re.sub(r'(?<=[^\W\d_]{3})[.,;:](?=[^\W\d_]{3})', ' ', q)     # "harry.potter" (not "j.k." / "node.js")
    out = []
    for tok in q.split():
        if INITIALS.fullmatch(tok):                          # "j.k." "r.l.": initials keep their dots
            out.append(tok)
            continue
        t = tok.strip('!?.,;:"“”()[]{}*~^<>|\\')
        t = re.sub(r'^[@$#]+(?=\w)', '', t)                  # "$potter", "#harrypotter"
        if re.search(r'[@$]', t) or (re.search(r'\d', t) and re.search(r'[^\W\d_]', t)):
            low = fold(t)
            if not (low in sys['vocab'] or len(docs_with(sys, low))):
                spellings = ['']
                for c in low:
                    subs = LOOKALIKES.get(c, c)
                    subs = subs if isinstance(subs, list) else [subs]
                    spellings = [s + x for s in spellings for x in subs][:8]
                for cand in spellings:
                    if len(cand) >= 4 and cand.isalpha() and (cand in sys['vocab'] or len(docs_with(sys, cand))):
                        t = cand
                        break
        if t:
            out.append(t)
    return ' '.join(out) or q


def search_books(query, sys, mood_filter=None, category_filter=None, top_k=24, free_only=False, online=True):
    """Public entry point. Surnames with an apostrophe ("o'connell") are tried as typed first; when the top
    authors don't have that surname, the joined spelling the catalog uses ("oconnell") is tried instead."""
    query = tidy_query(query, sys)
    out = _search_with_characters(query, sys, mood_filter, category_filter, top_k, free_only, online)
    joined = join_apostrophe_names(query, sys)
    if joined == query:
        return out
    surnames = [fold(m.group(1) + m.group(2)) for m in re.finditer(r"\b([od])['’]([a-zA-Z]{3,})\b", query, flags=re.I)]

    def hits(df):
        authors = [fold(a).replace("'", '') for a in df['author'].fillna('').head(3)] if 'author' in df.columns else []
        return sum(1 for s in surnames for a in authors if s in a)

    if hits(out[0]) >= 2:
        return out
    alt = _search_with_characters(joined, sys, mood_filter, category_filter, top_k, free_only, online)
    return alt if hits(alt[0]) > hits(out[0]) else out


def _search_with_characters(query, sys, mood_filter=None, category_filter=None, top_k=24, free_only=False, online=True):
    """The search below, plus famous characters ("hermione" -> Harry Potter books)
    and 'by <author> about <topic>' sentences."""
    typed = query
    at = None if mood_filter else author_topic(query, sys, category_filter, free_only, top_k)
    if at:
        return add_why(at[0], query, at[1], category_filter), at[1]
    df, banner = _search_books(query, sys, mood_filter, category_filter, top_k, free_only, online)
    df = exact_title_first(df, query)
    query = typed                                  # characters ("d'arcy") are matched on what was typed
    ch = character_series(query)
    if not ch or fold(ch[1]) == fold(query or '').strip():
        return df, banner
    name, series = ch
    sdf, _ = _search_books(series, sys, mood_filter, category_filter, top_k, free_only, False)
    if 'dedup_key' in sdf.columns and 'dedup_key' in df.columns:
        top = set(df['dedup_key'].head(8))
        if len(top & set(sdf['dedup_key'].head(12))) >= 3:
            return df, banner                     # the page already leads with the series ("sherlock")
        sdf = sdf[~sdf['dedup_key'].isin(top)]
    sdf = sdf.head(12).copy()
    if not len(sdf):
        return df, banner
    sdf['section'] = f"{name} is a character in {series}"
    attrs = dict(df.attrs)
    # a famous character's name: the series is what was meant ("hermione": Harry Potter, not Hermione Lee;
    # "katniss": Hunger Games, not books about kissing). What the words alone found stays below it.
    if len(df) and ('section' not in df.columns or not df['section'].fillna('').astype(bool).any()):
        df = df.copy()
        df['section'] = f"Results for “{(query or '').strip()}”"
    out = pd.concat([sdf, df.head(12)], ignore_index=True)
    banner = f"{name} is a character in {series}: showing those books first."
    out.attrs.update(attrs)
    return out, banner


def _search_books(query, sys, mood_filter=None, category_filter=None, top_k=24, free_only=False, online=True):
    """Public entry point: tidy the query, expand short forms, run the search, then attach the 'why' chips."""
    query = re.sub(r'(?<=[^\W\d_])[-_/]+(?=[^\W\d_])', ' ', query or '')   # "harry-potter" -> "harry potter"
    had_isbn_word = bool(ISBN_WORD.match(query))
    # "ISBN: 978-0-590-35342-7", "ISBN-13: 978...", but also "isbn 1338878921" (an ISBN starting with 13)
    tries = [re.sub(r'[\s-]', '', ISBN_WORD.sub('', query))]
    if ISBN_LABEL.match(query):
        tries.append(re.sub(r'[\s-]', '', ISBN_LABEL.sub('', query)))
    digits = next((d for d in tries if re.fullmatch(r'\d{9}[\dXx]|\d{13}', d)), tries[0])
    if re.fullmatch(r'\d{9}[\dXx]|\d{13}', digits):                          # an ISBN
        df, banner = isbn_search(digits, sys)
        return add_why(df, query, banner, category_filter), banner
    if re.fullmatch(r'\d{8,14}[Xx]?', digits) and (had_isbn_word or len(digits) >= 9):     # a mistyped ISBN
        banner = (f"“{digits}” has {len(digits)} digits, but an ISBN has 10 or 13. "
                  "A digit may be missing or extra: check the number on the book.")
        empty = sys['work_dedup'].iloc[0:0]
        return add_why(empty, query, banner, category_filter), banner
    if had_isbn_word:                             # "isbn of harry potter": the word itself is not a search word
        query = ISBN_WORD.sub('', query) or query
    pen = pen_name_of(query)
    if pen:                                       # "eric blair" -> George Orwell's books
        df, inner = search_books(pen[0], sys, mood_filter, category_filter, top_k, free_only, online)
        banner = f"{pen[1]} wrote as {pen[0].title()}: showing {pen[0].title()}'s books."
        if len(df) == 0 and inner:                # keep the reason ("No free books by ...")
            banner = f"{pen[1]} wrote as {pen[0].title()}. {inner}"
        return df, banner
    short = fold(query or '').strip().strip('.')
    expanded = SHORT_FORMS.get(short)
    if not expanded:
        # students' short words inside a longer query ("advance engineering maths", "intro to stats")
        words = query.split()
        long_words = [WORD_FORMS.get(fold(w), w) for w in words]
        if long_words != words:
            query = ' '.join(long_words)
    q = expanded or query

    # Misspelled words can mean several things ("poter" = potter OR peter): search the likeliest
    # meaning, then add the other meanings as their own sections and as "Did you mean" links.
    # slips=True: a key next to the right one ("fatsby" = gatsby) is always a possible first letter
    alternatives = [] if expanded else spelling_alternatives(sys, q, slips=True)
    if not alternatives and not expanded:
        done = phrase_completion(sys, q)                 # half typed: "harry p" -> harry potter
        if done:
            alternatives = [done]
    if alternatives:
        # the best guess may itself be one step from a far better-read phrase ("harry poer" -> "harry power" -> harry potter)
        better = [p for p in phrase_suggestions(sys, alternatives[0]) if p not in alternatives]
        alternatives = better[:2] + alternatives
        kw = dict(mood_filter=mood_filter, category_filter=category_filter, free_only=free_only)
        df, banner = _search_core(alternatives[0], sys, top_k=top_k, online=online, **kw)
        if "isn't in our library" in (banner or '') or (banner or '').startswith('No books match'):
            # last resort: the FIRST letter may be a slipped key ("orcy jackso" = percy jackson)
            # skip readings already tried FIRST, then take two ("orcy jackso": orczy and orc were tried,
            # percy jackson is the third slip reading)
            for alt in [a for a in spelling_alternatives(sys, q, slips=True) if a not in alternatives][:2]:
                d2, b2 = _search_core(alt, sys, top_k=top_k, online=online, **kw)
                if len(d2) and not ("isn't in our library" in (b2 or '') or (b2 or '').startswith('No books match')):
                    alternatives = [alt] + alternatives
                    df, banner = d2, b2
                    break
        if "isn't in our library" in (banner or '') or (banner or '').startswith('No books match'):
            for i, alt in enumerate(alternatives[1:], 1):
                d2, b2 = _search_core(alt, sys, top_k=top_k, online=online, **kw)
                if len(d2) and not ("isn't in our library" in (b2 or '') or (b2 or '').startswith('No books match')):
                    alternatives = [alt] + alternatives[:i] + alternatives[i + 1:]
                    df, banner = d2, b2
                    break
        if (("isn't in our library" in (banner or '') or (banner or '').startswith('No books match'))
                and not books_with_all_words(sys, alternatives[0])):
            alternatives = []        # a made-up reading that no book has ("agatha hartini"): the page then explains the typed words
        if banner and banner.startswith('No books match'):
            banner = None
        shown = set(df['dedup_key']) if 'dedup_key' in df.columns else set()
        extra = []
        live = alternatives[:1]
        for alt in alternatives[1:]:
            d2, b2 = _search_core(alt, sys, top_k=8, online=False, **kw)
            if not len(d2) or "isn't in our library" in (b2 or '') or (b2 or '').startswith('No books match'):
                continue                                    # a link must never lead to an empty page
            live.append(alt)
            if 'dedup_key' in d2.columns:
                d2 = d2[~d2['dedup_key'].isin(shown)]
            d2 = d2.head(6).copy()
            if len(d2):
                d2['section'] = f"Did you mean “{alt}”?"
                shown |= set(d2['dedup_key']) if 'dedup_key' in d2.columns else set()
                extra.append(d2)
        alternatives = live
        if extra:
            df = pd.concat([df] + extra, ignore_index=True)
        if len(df) == 0:
            alternatives = []                     # no meaning found anything: the normal path explains why
    if alternatives:
        note = '' if alternatives[0] == q else f"Showing results for “{alternatives[0]}”."
        kept = banner if banner and not banner.startswith(('Searched for', 'Did you mean')) else ''
        if kept.startswith('Showing results for author'):
            note = ''                                    # the author line already says what is shown
        banner = ' '.join(x for x in (note, kept) if x) or None
        df, banner = _add_mentions(sys, df, alternatives[0], banner, category_filter, free_only)
        out = add_why(df, alternatives[0], banner, category_filter)
        out.attrs['alternatives'] = alternatives
        return out, banner

    df, banner = _search_core(q, sys, mood_filter=mood_filter, category_filter=category_filter,
                              top_k=top_k, free_only=free_only, online=online)
    # Safety net: nothing found, but a title is a very close spelling match -> search that title instead
    if len(df) == 0 and q and q.strip() and (banner or '').startswith('No books match'):
        tb = closest_title(sys, q, cutoff=88)
        if tb:
            title = sys['title_list'][tb[2]]
            df2, _ = _search_core(title, sys, mood_filter=mood_filter, category_filter=category_filter,
                                  top_k=top_k, free_only=free_only, online=online)
            if len(df2):
                df, banner, q = df2, f"Did you mean “{title}”? Showing results for it.", title
    if expanded and (not banner or banner.startswith('Searched for')):
        banner = f"Showing results for “{expanded}” ({short.upper()})"

    # "percy grey" -> Zane Grey: the surname fits, the first name does not. Say so, and offer the other meanings.
    misfit = []
    if (banner or '').startswith('Showing results for author: ') and not expanded:
        name = banner[len('Showing results for author: '):]
        name_words = proc(name).split()
        for t in q.split():
            w = fold(t).strip('.,!?;:"()')
            # a real word (not a typo) that is like no part of the name, and no initial stands for it
            # ("kathleen rowling" = J. K. Rowling, "ronald tolkien" = J. R. R. Tolkien fit)
            if (w.isalpha() and len(w) >= 3 and w not in STOPWORDS and name_words and len(docs_with(sys, w))
                    and max(fuzz.ratio(w, n) for n in name_words) < 70
                    and not any(len(n) <= 3 and w[0] in n for n in name_words)):
                misfit.append(w)
        if misfit and len(misfit) < len(q.split()):
            banner = f"No author named “{q.strip().title()}”. Showing {name}, the closest name."

    # What was typed stays first. Below it, offer other likely meanings as links and small sections:
    #  - a real word that is a likely slip for a far more read one ("hobbs" -> hobbit, hobby)
    #  - a very misspelled query that is close to a whole title or author name ("shrlck hlms")
    suggestions = [] if expanded else popular_neighbours(sys, q)
    if not expanded:                                  # a whole phrase one step from a far better-read one ("harry power")
        suggestions = list(dict.fromkeys(AMBIGUOUS_SHORT.get(short, []) + suggestions + phrase_suggestions(sys, q)))[:3]
    if misfit and len(misfit) < len(q.split()):
        suggestions = list(dict.fromkeys(misfit + suggestions))[:3]     # "percy" on its own is a meaning too
    weak = len(df) == 0 or (banner or '').startswith(('No books match', '“'))
    unknown_word = any(w.isalpha() and len(w) >= 3 and w not in STOPWORDS and not len(docs_with(sys, w))
                       for w in (fold(t).strip('.,!?;:"()') for t in q.split()))
    guessed = (banner or '').startswith(('Searched for', 'Did you mean'))   # the search had to guess a fix
    author_found = (banner or '').startswith('Showing results for author')   # the typo already became a real author
    if not suggestions and not expanded and unknown_word and (weak or guessed) and not author_found:
        suggestions = phrase_guesses(sys, q)
    # never suggest what the page already shows ("stfen kng" already shows Stephen King)
    suggestions = [s for s in suggestions if fold(s) not in fold(banner or '') or s in misfit]
    if suggestions:
        kw = dict(mood_filter=mood_filter, category_filter=category_filter, free_only=free_only)
        shown = set(df['dedup_key']) if 'dedup_key' in df.columns else set()
        extra = []
        live_links = []
        for alt in suggestions:
            d2, b2 = _search_core(alt, sys, top_k=8, online=False, **kw)
            if not len(d2) or "isn't in our library" in (b2 or '') or (b2 or '').startswith('No books match'):
                continue                                    # a link must never lead to an empty page
            live_links.append(alt)
            if 'dedup_key' in d2.columns:
                d2 = d2[~d2['dedup_key'].isin(shown)]
                shown |= set(d2['dedup_key'].head(6))
            d2 = d2.head(6).copy()
            if len(d2):
                d2['section'] = f"Did you mean “{alt}”?"
                extra.append(d2)
        suggestions = live_links
        if extra:
            if len(df) and ('section' not in df.columns or not df['section'].fillna('').astype(bool).any()):
                df = df.copy()
                df['section'] = f"Results for “{q.strip()}”"
            reads = lambda d: float(pd.to_numeric(d['rating_count'], errors='coerce').fillna(0).head(5).sum()) \
                if 'rating_count' in d.columns else 0.0
            if guessed and len(df) and reads(extra[0]) >= 10 * max(reads(df), 1):
                # the search only GUESSED its fix ("lrd of th rngs" -> Lord of the Ringards) and a suggestion
                # is far more read (The Lord of the Rings): show the suggestion first
                df = pd.concat([extra[0], df] + extra[1:], ignore_index=True)
                banner = f"Showing results for “{suggestions[0]}”."
            else:
                df = pd.concat([df] + extra, ignore_index=True)
            if len(extra) and (banner or '').startswith('No books match'):
                banner = f"No exact match for “{q.strip()}”. These are the closest guesses."
                # ("“…” isn't in our library yet" stays as it is: it is the clearer message)
    if not expanded:
        df, banner = _add_mentions(sys, df, q, banner, category_filter, free_only)
    out = add_why(df, q, banner, category_filter)
    if suggestions:
        out.attrs['suggestions'] = suggestions
    return out, banner

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
