"""'Readers who liked this also liked': from the anonymous Book-Crossing ratings (ratings_clean.parquet).

Privacy: only COUNTS across readers are used. A book is suggested only when at least MIN_READERS different
readers liked both books, so no single reader's list can ever be seen or reconstructed.
Built once in the background at startup (a few seconds); until then it simply returns nothing.
"""
import os
import threading

import numpy as np
import pandas as pd

from search import to_isbn13

LIKED = 7            # a rating of 7-10 out of 10 counts as "liked"
MIN_READERS = 3      # at least this many readers in common, for privacy and to avoid chance matches
MAX_READERS = 3000   # very popular books: a sample of their readers is plenty

_state = {}
_lock = threading.Lock()


def build(sys, data_dir):
    """Map ratings to our books (by ISBN -> work) and keep, per book, who liked it and, per reader, what."""
    path = os.path.join(data_dir, 'ratings_clean.parquet')
    if not os.path.exists(path):
        return
    r = pd.read_parquet(path, columns=['User-ID', 'ISBN', 'Book-Rating'])
    r = r[r['Book-Rating'] >= LIKED]
    isbn13 = r['ISBN'].astype(str).map(lambda x: to_isbn13(x.upper()) or x)
    books = sys['books'][['isbn13', 'dedup_key']].dropna().drop_duplicates('isbn13')
    key_of_isbn = dict(zip(books['isbn13'], books['dedup_key']))
    keys = isbn13.map(key_of_isbn)
    r = pd.DataFrame({'user': r['User-ID'].to_numpy(), 'key': keys.to_numpy()}).dropna().drop_duplicates()
    if not len(r):
        return
    key_codes, key_names = pd.factorize(r['key'])
    user_codes, _ = pd.factorize(r['user'])
    order_k = np.argsort(key_codes, kind='stable')
    order_u = np.argsort(user_codes, kind='stable')
    k_start = np.searchsorted(key_codes[order_k], np.arange(len(key_names) + 1))
    u_start = np.searchsorted(user_codes[order_u], np.arange(user_codes.max() + 2))
    with _lock:
        _state.update(
            key_index={k: i for i, k in enumerate(key_names)}, key_names=np.asarray(key_names),
            users_of=(user_codes[order_k], k_start),          # book -> readers who liked it
            books_of=(key_codes[order_u], u_start),           # reader -> books they liked
            likers=np.bincount(key_codes, minlength=len(key_names)),
        )


def start(sys, data_dir):
    threading.Thread(target=build, args=(sys, data_dir), daemon=True).start()


def also_liked(key, limit=6):
    """-> list of dedup_keys of books liked by the same readers, best first (empty if not enough readers)."""
    with _lock:
        s = dict(_state)
    i = s.get('key_index', {}).get(key) if s else None
    if i is None:
        return []
    users_flat, k_start = s['users_of']
    readers = users_flat[k_start[i]:k_start[i + 1]]
    if len(readers) < MIN_READERS:
        return []
    if len(readers) > MAX_READERS:
        readers = np.random.default_rng(0).choice(readers, MAX_READERS, replace=False)
    books_flat, u_start = s['books_of']
    other = np.concatenate([books_flat[u_start[u]:u_start[u + 1]] for u in readers])
    counts = np.bincount(other, minlength=len(s['key_names']))
    counts[i] = 0
    # a much-read book needs more readers in common than 3 (3 of 900 readers is coincidence): at least 2%
    need = max(MIN_READERS, int(0.02 * len(readers)))
    ok = np.flatnonzero(counts >= need)
    if not len(ok):
        return []
    # shared readers relative to both books' readers (Jaccard): favours real overlap, not tiny or huge books
    score = counts[ok] / (len(readers) + s['likers'][ok] - counts[ok])
    best = ok[np.argsort(-score, kind='stable')[:limit]]
    return [str(s['key_names'][b]) for b in best]
