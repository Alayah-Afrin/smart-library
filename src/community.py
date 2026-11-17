"""Community ratings + comments (no login).

Each browser gets a random anonymous id (a cookie). It is only a vote limiter: one rating per
book per browser. Kept apart from the Book-Crossing ratings so the evaluation data stays untouched.
"""
import os
import sqlite3
import threading

DB_PATH = os.path.join('instance', 'community.sqlite3')
_lock = threading.Lock()


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init():
    with _connect() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS reviews (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                book_key   TEXT NOT NULL,
                anon_id    TEXT NOT NULL,
                rating     INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment    TEXT NOT NULL DEFAULT '',
                reports    INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (book_key, anon_id)
            );
            CREATE INDEX IF NOT EXISTS reviews_book ON reviews(book_key);
            CREATE TABLE IF NOT EXISTS reports (
                review_id  INTEGER NOT NULL,
                anon_id    TEXT NOT NULL,
                reason     TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (review_id, anon_id)
            );
        """)
        cols = {r['name'] for r in con.execute('PRAGMA table_info(reviews)')}
        if 'sentiment' not in cols:                                 # upgrade older databases
            con.execute('ALTER TABLE reviews ADD COLUMN sentiment TEXT')
            con.execute('ALTER TABLE reviews ADD COLUMN sentiment_score REAL')
        if 'checked' not in cols:                                   # moderator looked at it and kept it
            con.execute('ALTER TABLE reviews ADD COLUMN checked INTEGER NOT NULL DEFAULT 0')
            con.execute("UPDATE reviews SET checked = 1, sentiment = NULL WHERE sentiment = 'checked'")


def summary(book_key, anon_id='', limit=30):
    with _connect() as con:
        row = con.execute('SELECT AVG(rating) avg, COUNT(*) n FROM reviews WHERE book_key = ?', (book_key,)).fetchone()
        comments = con.execute(
            "SELECT id, rating, comment, created_at, sentiment FROM reviews WHERE book_key = ? AND comment != '' AND reports < 3 "
            "ORDER BY created_at DESC LIMIT ?", (book_key, limit)).fetchall()
        mine = con.execute('SELECT rating, comment FROM reviews WHERE book_key = ? AND anon_id = ?',
                           (book_key, anon_id)).fetchone() if anon_id else None
        feel = {r['sentiment']: r['n'] for r in con.execute(
            "SELECT sentiment, COUNT(*) n FROM reviews WHERE book_key = ? AND sentiment IS NOT NULL AND reports < 3 "
            "GROUP BY sentiment", (book_key,))}
    return {
        'avg': round(row['avg'], 2) if row['avg'] is not None else None,
        'count': row['n'],
        'comments': [dict(c) for c in comments],
        'mine': dict(mine) if mine else None,
        'sentiment': sentiment_summary(feel),
    }


def sentiment_summary(feel):
    """{'positive': 3, 'negative': 1, ...} -> counts, percentages and a one-line verdict (or None)."""
    feel = {k: v for k, v in feel.items() if k in ('positive', 'neutral', 'negative')}   # skip 'checked'
    total = sum(feel.values())
    if not total:
        return None
    pct = {k: round(100 * feel.get(k, 0) / total) for k in ('positive', 'neutral', 'negative')}
    if pct['positive'] >= 65:
        verdict = 'Mostly positive'
    elif pct['negative'] >= 65:
        verdict = 'Mostly negative'
    elif pct['neutral'] >= 50:
        verdict = 'Mostly neutral'
    else:
        verdict = 'Mixed opinions'          # e.g. 50% positive / 50% negative
    return {'total': total, 'counts': {k: feel.get(k, 0) for k in pct}, 'percent': pct, 'verdict': verdict}


def has_reviews(book_key):
    with _connect() as con:
        return con.execute('SELECT 1 FROM reviews WHERE book_key = ? LIMIT 1', (book_key,)).fetchone() is not None


def averages(book_keys):
    """book_key -> (avg, count) for many books at once (for the result cards)."""
    keys = [k for k in dict.fromkeys(book_keys) if k]
    if not keys:
        return {}
    out = {}
    with _connect() as con:
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = f"SELECT book_key, AVG(rating) a, COUNT(*) n FROM reviews WHERE book_key IN ({','.join('?' * len(chunk))}) GROUP BY book_key"
            for r in con.execute(q, chunk):
                out[r['book_key']] = (round(r['a'], 1), r['n'])
    return out


def save(book_key, anon_id, rating, comment, sentiment=None, sentiment_score=None):
    with _lock, _connect() as con:
        con.execute("""
            INSERT INTO reviews (book_key, anon_id, rating, comment, sentiment, sentiment_score) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (book_key, anon_id) DO UPDATE SET
                rating = excluded.rating, comment = excluded.comment,
                sentiment = excluded.sentiment, sentiment_score = excluded.sentiment_score,
                reports = 0, checked = 0, created_at = CURRENT_TIMESTAMP
        """, (book_key, anon_id, rating, comment, sentiment, sentiment_score))


def missing_sentiment(limit=500):
    """Comments saved while the model wasn't loaded yet."""
    with _connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, comment FROM reviews WHERE sentiment IS NULL AND comment != '' LIMIT ?", (limit,))]


def set_sentiment(rows):
    """rows: [(review_id, label, score), ...]"""
    with _lock, _connect() as con:
        con.executemany('UPDATE reviews SET sentiment = ?, sentiment_score = ? WHERE id = ?',
                        [(lab, sc, rid) for rid, lab, sc in rows])


def needs_review(limit=100):
    """For moderators: reported comments, plus comments whose stars and words disagree."""
    with _connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT reviews.*,
                   (SELECT GROUP_CONCAT(reason) FROM reports WHERE reports.review_id = reviews.id) AS report_reasons
            FROM reviews
            WHERE reports >= 1
               OR (checked = 0 AND rating >= 4 AND sentiment = 'negative')
               OR (checked = 0 AND rating <= 2 AND sentiment = 'positive')
            ORDER BY reports DESC, created_at DESC LIMIT ?""", (limit,))]


def site_sentiment():
    """Whole-site numbers for the moderation page."""
    with _connect() as con:
        feel = {r['sentiment']: r['n'] for r in con.execute(
            'SELECT sentiment, COUNT(*) n FROM reviews WHERE sentiment IS NOT NULL GROUP BY sentiment')}
    return sentiment_summary(feel)


REPORT_REASONS = {'spam': 'Spam', 'offensive': 'Offensive', 'spoiler': 'Spoiler', 'other': 'Other'}


def report(review_id, anon_id, reason):
    """One report per comment per browser, with a reason. Returns False if this browser already reported it."""
    with _lock, _connect() as con:
        if con.execute('SELECT 1 FROM reviews WHERE id = ?', (review_id,)).fetchone() is None:
            return False
        cur = con.execute('INSERT OR IGNORE INTO reports (review_id, anon_id, reason) VALUES (?, ?, ?)',
                          (review_id, anon_id, reason))
        if cur.rowcount:
            con.execute('UPDATE reviews SET reports = reports + 1 WHERE id = ?', (review_id,))
        return bool(cur.rowcount)


def reported(min_reports=1):
    with _connect() as con:
        return [dict(r) for r in con.execute(
            'SELECT * FROM reviews WHERE reports >= ? ORDER BY reports DESC, created_at DESC', (min_reports,))]


def delete(review_id):
    with _lock, _connect() as con:
        con.execute('DELETE FROM reviews WHERE id = ?', (review_id,))
        con.execute('DELETE FROM reports WHERE review_id = ?', (review_id,))


def clear_reports(review_id):
    """'Keep' in moderation: clear reports and mark the stars/words mismatch as checked."""
    with _lock, _connect() as con:
        con.execute('UPDATE reviews SET reports = 0, checked = 1 WHERE id = ?', (review_id,))
        con.execute('DELETE FROM reports WHERE review_id = ?', (review_id,))
