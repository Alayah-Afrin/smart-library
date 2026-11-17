"""Tests for the free filter, in-site reader, community reviews, shelf and admin pages.
Run from D:/documents/Smart_Library_V2:   python tests/test_features.py
Uses a temporary review database, so the real one is never touched."""
import os
import re
import sys
import tempfile

sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import community
community.DB_PATH = os.path.join(tempfile.mkdtemp(prefix='sl-test-'), 'community.sqlite3')
import search as S
S.live_lookup = lambda *a, **k: []                         # no internet for search during tests
S.get_better_cover = lambda isbn, title, author, fallback: fallback

import app as A
import reader

c = A.app.test_client()
passed = []


def check(name, fn):
    try:
        ok, info = fn()
    except Exception as e:
        ok, info = False, f'CRASH {type(e).__name__}: {e}'
    passed.append(ok)
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'   -> {info}'))


def search(q, **kw):
    df, banner = S.search_books(q, A.system, top_k=24, **kw)
    return df, banner or ''


# ---- free / category / mood filters on author searches ----
for q in ['rowling', 'tolkien', 'jane austen', 'charles dickens', 'mark twain']:
    def f(q=q):
        df, b = search(q, free_only=True)
        bad = df[(df.get('read_url', '') == '')] if len(df) else df
        return len(bad) == 0, f'{len(bad)} non-free of {len(df)}; banner={b!r}'
    check(f'free only + author "{q}" -> only free books', f)
check('free only + "rowling" says no free books',
      lambda: (lambda d, b: (len(d) == 0 and 'No free books by' in b, (len(d), b)))(*search('rowling', free_only=True)))
check('free only + "dickens" still finds free Dickens',
      lambda: (lambda d, b: (len(d) > 0, b))(*search('charles dickens', free_only=True)))
check('category filter applies to author search',
      lambda: (lambda d, b: (all('computer' in str(x).lower() or 'programming' in str(x).lower() or 'software' in str(x).lower()
                                 for x in d['categories']), b))(*search('stephen king', category_filter='Computer Science')))
check('mood filter applies to author search',
      lambda: (lambda d, b: (len(d) == 0 or set(d['mood']) == {'fear'}, set(d['mood']) if len(d) else b))(*search('stephen king', mood_filter='fear')))
check('no filters: author search unchanged',
      lambda: (lambda d, b: (len(d) > 0 and 'Rowling' in b, b))(*search('rowling')))

# ---- reader ----
SAMPLE = ("Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\nCHAPTER I.\n\n"
          + "\n\n".join(f"Paragraph {i} of the test book." for i in range(1, 121))
          + "\n\nCHAPTER II.\n\nLast line.\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicence text")
check('clean() drops header and footer',
      lambda: (lambda t: ('Header junk' not in t and 'licence' not in t and t.startswith('CHAPTER I.'), t[:40]))(reader.clean(SAMPLE)))
check('mirror folder layout', lambda: ((reader.mirror_dir(1342), reader.mirror_dir(11), reader.mirror_dir(5))
                                       == ('1/3/4/1342', '1/11', '0/5'), (reader.mirror_dir(1342), reader.mirror_dir(11), reader.mirror_dir(5))))
check('gutenberg.org first, then mirrors', lambda: (lambda u: ('www.gutenberg.org' in u[0][0] and 'xmission' in u[2][0]
                                                               and u[2][0].endswith('/1/3/4/1342/1342-0.txt'), [x[0] for x in u[:3]]))(reader.candidate_urls(1342)))


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self.content = code, body


def fake_get(url, headers=None, timeout=None):
    if 'www.gutenberg.org' in url:
        return FakeResp(503, b'<html>Temporarily Down for Maintenance</html>')
    if 'xmission' in url and url.endswith('-0.txt'):
        return FakeResp(200, b'<html>404 page sent with status 200</html>')     # html error page must be skipped
    if 'csclub' in url and url.endswith('-8.txt'):
        return FakeResp(200, 'Caf\xe9 text in latin-1'.encode('latin-1'))
    return FakeResp(404, b'')


real_requests_get = reader.requests.get
reader.requests.get = fake_get
check('main site down -> falls back to a mirror (skips HTML, decodes latin-1)',
      lambda: (lambda t: (t == 'Café text in latin-1', t))(reader._download(1342, 'test')))
reader.requests.get = lambda *a, **k: FakeResp(503, b'down')


def all_down():
    try:
        reader._download(1342, 'test')
        return False, 'no error raised'
    except reader.ReaderError:
        return True, ''


check('every source down -> ReaderError', all_down)
reader.requests.get = real_requests_get

gid = next(iter(A.READABLE))
real_get_text = reader.get_text
reader.get_text = lambda g, ua: SAMPLE
A._book_pages.cache_clear()
check(f'/read/{gid} page 1 renders', lambda: (lambda r: (r.status_code == 200 and b'Paragraph 1 of' in r.data and b'Page 1 of' in r.data, r.status_code))(c.get(f'/read/{gid}')))
check('page 2 renders', lambda: (lambda r: (r.status_code == 200 and b'Page 2 of' in r.data, r.status_code))(c.get(f'/read/{gid}?page=2')))
check('page 999 clamps to last page', lambda: (lambda r: (r.status_code == 200 and b'Last line.' in r.data, r.status_code))(c.get(f'/read/{gid}?page=999')))
check('page=-5 clamps to first', lambda: (lambda r: (b'Page 1 of' in r.data, r.status_code))(c.get(f'/read/{gid}?page=-5')))
check('book not in catalog -> 404', lambda: (c.get('/read/99999999').status_code == 404, ''))


def offline(g, ua):
    raise reader.ReaderError()


reader.get_text = offline
A._book_pages.cache_clear()
check('Gutenberg down -> friendly 503 page', lambda: (lambda r: (r.status_code == 503 and b"isn't answering" in r.data, r.status_code))(c.get(f'/read/{gid}')))
reader.get_text = real_get_text
A._book_pages.cache_clear()

# ---- community ----
home = c.get('/').get_data(as_text=True)
token = re.search(r'name="csrf-token" content="([^"]+)"', home).group(1)
key = A.system['work_dedup']['dedup_key'].iloc[0]
H = {'X-CSRF-Token': token}
check('GET reviews needs a known book', lambda: (c.get('/api/community?key=nope').status_code == 400, ''))
check('GET reviews for a book', lambda: (lambda j: (j['count'] == 0 and j['comments'] == [], j))(c.get('/api/community', query_string={'key': key}).get_json()))
check('POST without token is rejected', lambda: (c.post('/api/community', json={'key': key, 'rating': 5}).status_code == 400, ''))
check('POST rating 7 is rejected', lambda: (c.post('/api/community', json={'key': key, 'rating': 7}, headers=H).status_code == 400, ''))
check('POST rating + comment saved', lambda: (lambda j: (j['count'] == 1 and j['avg'] == 4 and j['comments'][0]['comment'] == 'Lovely <b>book</b>', j))(
    c.post('/api/community', json={'key': key, 'rating': 4, 'comment': 'Lovely <b>book</b>'}, headers=H).get_json()))
check('same browser updates instead of voting twice', lambda: (lambda j: (j['count'] == 1 and j['avg'] == 2, j))(
    c.post('/api/community', json={'key': key, 'rating': 2, 'comment': 'changed my mind'}, headers=H).get_json()))
c2 = A.app.test_client()
t2 = re.search(r'name="csrf-token" content="([^"]+)"', c2.get('/').get_data(as_text=True)).group(1)
check('second browser adds a second vote', lambda: (lambda j: (j['count'] == 2 and j['avg'] == 3.5, j))(
    c2.post('/api/community', json={'key': key, 'rating': 5}, headers={'X-CSRF-Token': t2}).get_json()))
rid = community.summary(key)['comments'][0]['id']
check('report without a reason is refused', lambda: (c.post(f'/api/community/report/{rid}', json={}, headers=H).status_code == 400, ''))
check('report with a made-up reason is refused', lambda: (c.post(f'/api/community/report/{rid}', json={'reason': 'boring'}, headers=H).status_code == 400, ''))
check('report with a reason counts once per browser', lambda: (
    c.post(f'/api/community/report/{rid}', json={'reason': 'spam'}, headers=H).get_json() == {'ok': True, 'already': False}
    and c.post(f'/api/community/report/{rid}', json={'reason': 'spam'}, headers=H).get_json() == {'ok': True, 'already': True}
    and community.reported()[0]['reports'] == 1, community.reported()))
check('admin list shows the report reason', lambda: (any(r.get('report_reasons') == 'spam' for r in community.needs_review()), community.needs_review()[:1]))
check('card shows community rating', lambda: (lambda d: (d[key] == (3.5, 2), d))(community.averages([key])))

# ---- sentiment of comments ----
import sentiment
model_ok = sentiment.load_now()
check(f'sentiment model loads ({sentiment.status()})', lambda: (model_ok, sentiment.status()))
if model_ok:
    cases = [('Absolutely loved it, could not put it down!', 'positive'), ('Boring and way too long.', 'negative'),
             ('meh', 'neutral'), ('It was okay I guess.', 'neutral'), ('The ending ruined it for me', 'negative'),
             ('Great book for beginners in calculus', 'positive')]
    got = sentiment.analyse([t for t, _ in cases])
    for (text, want), (lab, sc) in zip(cases, got):
        check(f'sentiment "{text}" -> {want}', lambda lab=lab, want=want, sc=sc: (lab == want, f'{lab} {sc}'))
    check('empty comment -> no sentiment', lambda: (sentiment.analyse([''])[0] == (None, None), sentiment.analyse([''])))

    key2 = A.system['work_dedup']['dedup_key'].iloc[1]
    check('posted comment gets its sentiment', lambda: (lambda j: (j['my_sentiment'] == 'positive' and j['comments'][0]['sentiment'] == 'positive'
                                                                   and j['sentiment']['verdict'] == 'Mostly positive', j))(
        c.post('/api/community', json={'key': key2, 'rating': 5, 'comment': 'Wonderful, I loved every chapter'}, headers=H).get_json()))
    check('5 stars + angry words is flagged', lambda: (lambda j: (j['stars_words_disagree'] is True, j))(
        c2.post('/api/community', json={'key': key2, 'rating': 5, 'comment': 'Terrible, a complete waste of my time'},
                headers={'X-CSRF-Token': t2}).get_json()))
    check('mixed opinions verdict', lambda: (lambda s: (s['sentiment']['verdict'] == 'Mixed opinions' and s['sentiment']['percent'] == {'positive': 50, 'neutral': 0, 'negative': 50}, s['sentiment']))(
        community.summary(key2)))
    check('flagged review shows up for moderators', lambda: (any(r['comment'].startswith('Terrible') for r in community.needs_review()), ''))

    # a comment saved while the model was still loading gets analysed later
    community.save(A.system['work_dedup']['dedup_key'].iloc[2], 'late-browser', 1, 'Awful book, I hated it')
    A.backfill_sentiment()
    check('backfill analyses older comments', lambda: (lambda s: (s['comments'][0]['sentiment'] == 'negative', s['comments']))(
        community.summary(A.system['work_dedup']['dedup_key'].iloc[2])))

# ---- pages ----
check('/shelf renders', lambda: (c.get('/shelf').status_code == 200, ''))
check('/admin is off without ADMIN_PASSWORD', lambda: (c.get('/admin').status_code == 404, ''))
A.ADMIN_PASSWORD = 'test-admin-pass'
check('/admin wrong password stays locked', lambda: (b'Wrong password' in c.post('/admin', data={'csrf': token, 'password': 'x'}, follow_redirects=True).data, ''))
check('/admin delete removes reported comment', lambda: (
    c.post('/admin', data={'csrf': token, 'password': 'test-admin-pass'}).status_code == 302
    and c.post('/admin', data={'csrf': token, 'id': rid, 'action': 'delete'}).status_code == 302
    and community.summary(key)['count'] == 1, community.summary(key)))
check('search page still works', lambda: (c.post('/', data={'query': 'harry potter'}).status_code == 200, ''))
check('search is a normal link (Back works): /?q=harry+potter', lambda: (lambda r: (
    r.status_code == 200 and 'value="harry potter"' in r.get_data(as_text=True) and 'class="pick"' not in r.get_data(as_text=True), r.status_code))(
    c.get('/?q=harry+potter')))
check('search form uses GET', lambda: ('method="GET"' in c.get('/').get_data(as_text=True), ''))
check('category + unrelated query says so (no random books)', lambda: (lambda df, b: (len(df) == 0 and b.startswith('No Mathematics books match'), b))(
    *search('harry potter', category_filter='Mathematics')))
check('free + category message', lambda: (lambda df, b: (len(df) == 0 and b.startswith('No free Fiction books match'), b))(
    *search('linear algebra', category_filter='Fiction', free_only=True)))
check('gibberish -> clear "no books match" (no random books)', lambda: (lambda df, b: (len(df) == 0 and b.startswith('No books match'), b))(
    *search('asdfghjk')))
check('related category results still show (calculus + Mathematics)', lambda: (lambda df, b: (
    len(df) > 0 and any('calculus' in str(t).lower() for t in df['title'].head(5)), df['title'].head(5).tolist()))(
    *search('calculus', category_filter='Mathematics')))
home_html = c.get('/').get_data(as_text=True)
pick_part = home_html.split('class="pick"')[1].split('</main>')[0] if 'class="pick"' in home_html else ''
check("Today's read: no side-panel button, has Save", lambda: ('summary-btn' not in pick_part and 'save-btn' in pick_part, 'pick block wrong'))
check("Today's read never shows a subject list as a summary", lambda: ('class="pick-desc">Subjects:' not in home_html, 'subjects shown as summary'))

# ---- "harr": a real surname -> its books first, plus a "Did you mean" link ----
harr_html = c.get('/?q=harr').get_data(as_text=True)
check('"harr" shows books by the Harr authors', lambda: ('A Civil Action' in harr_html, 'no Jonathan Harr'))
check('"harr" offers a "Did you mean harry" link', lambda: ('href="/?q=harry"' in harr_html, 'no did-you-mean link'))
check('"harry pot" (not a name) still finds Harry Potter', lambda: (lambda d, b: ('Harry Potter' in str(d['title'].iloc[0]), d['title'].head(2).tolist()))(*search('harry pot')))

# ---- textbook PDFs inside our reader (fake PDF: no internet needed) ----
import pdfs
if A.PDF_BOOKS:
    pid = 0
    real_ensure = pdfs.ensure_pdf
    fake = os.path.join(tempfile.mkdtemp(prefix='sl-pdf-'), 'fake.pdf')
    open(fake, 'wb').write(b'%PDF-1.4\n%fake test pdf\n')
    pdfs.ensure_pdf = lambda url, ua: fake
    check('/read/pdf/0 page renders with the book inside', lambda: (lambda r: (r.status_code == 200 and b'/pdf/0' in r.data and b'pdf-frame' in r.data, r.status_code))(c.get('/read/pdf/0')))
    check('/pdf/0 serves a PDF', lambda: (lambda r: (r.status_code == 200 and r.mimetype == 'application/pdf' and r.data.startswith(b'%PDF'), (r.status_code, r.mimetype)))(c.get('/pdf/0')))
    def down(url, ua):
        raise pdfs.PdfError('down')
    pdfs.ensure_pdf = down
    check('publisher down -> friendly page, not a crash', lambda: (lambda r: (r.status_code == 502 and b"couldn't be opened here" in r.data, r.status_code))(c.get('/pdf/0')))
    pdfs.ensure_pdf = real_ensure
    check('PDF outside our list -> 404 (no open proxy)', lambda: (c.get(f'/read/pdf/{len(A.PDF_BOOKS)}').status_code == 404 and c.get('/pdf/99999').status_code == 404, ''))
    check('cache file name follows the address, not the list position', lambda: (pdfs.cached_path('https://a/x.pdf') == pdfs.cached_path('https://a/x.pdf') != pdfs.cached_path('https://a/y.pdf'), ''))
    rec = A.decorate([{'read_url': A.PDF_BOOKS[0]['read_url'], 'title': 't', 'why': ''}])[0]
    check('a textbook with a working PDF gets "Read here"', lambda: (rec['pdf_id'] == 0, rec.get('pdf_id')))

# ---- "Did you mean" picks the popular word; night mode is on every page ----
hobb_html = c.get('/?q=hobb').get_data(as_text=True)
check('"hobb" suggests hobbit (popular), not hobbies', lambda: ('href="/?q=hobbit"' in hobb_html, 'wrong suggestion'))
for path in ['/', '/?q=harry', '/top', '/shelf']:
    check(f'night-mode button + early theme script on {path}', lambda path=path: (lambda h: ('id="theme-btn"' in h and "sl_theme" in h, 'missing'))(c.get(path).get_data(as_text=True)))

# ---- spam guard for online books ----
check('rating a made-up online book is refused', lambda: (c.get('/api/community?key=ol:/works/OL99999999W').status_code == 400, ''))
A.SHOWN_ONLINE['ol:/works/OL12345W'] = True
check('rating an online book that was shown is allowed', lambda: (c.get('/api/community?key=ol:/works/OL12345W').status_code == 200, ''))
res_html = c.get('/?q=charles+dickens&free=1').get_data(as_text=True)
cards = res_html.split('<article class="book-card">')[1:]
check('each card has exactly one read button (no duplicate free link)', lambda: (
    len(cards) > 0 and all(card.count('class="btn-primary"') + card.count('class="btn-secondary"') == 1 for card in cards)
    and 'chip-link' not in res_html, f'{len(cards)} cards'))
check('textbook card says "Read free" (opens publisher site)', lambda: (lambda h: ('Read free ↗' in h, 'no Read free button'))(
    c.get('/?q=linear+algebra&free=1').get_data(as_text=True)))
# ---- regressions found by the wide query check ----
check('"jk" finds J.K. Rowling (initials)', lambda: (lambda d, b: (len(d) >= 10 and (d['author'].str.contains('Rowling').mean() > 0.8), d['author'].head(3).tolist()))(*search('jk')))
check('"jrr tolkien" finds Tolkien', lambda: (lambda d, b: ('tolkien' in b.lower(), b))(*search('jrr tolkien')))
check('"ai" means artificial intelligence', lambda: (lambda d, b: ('artificial intelligence' in b, b))(*search('ai')))
check('"introduction to physics" is NOT changed to metaphysics', lambda: (lambda d, b: ('metaphysics' not in b.lower() and 'metaphysics' not in str(d['title'].iloc[0]).lower(), (b, d['title'].head(2).tolist())))(*search('introduction to physics')))
check('"lotr" -> Lord of the Rings first', lambda: (lambda d, b: ('lord of the rings' in str(d['title'].iloc[0]).lower(), d['title'].head(2).tolist()))(*search('lotr')))
check('"shakspeare" -> Shakespeare', lambda: (lambda d, b: ('Shakespeare' in b, b))(*search('shakspeare')))
check('2-word search fills the page ("learn python")', lambda: (lambda d, b: (len(d) >= 10, len(d)))(*search('learn python')))
check('title we do not have says so ("project hail mary")', lambda: (lambda d, b: ("isn't in our library" in b, b))(*search('project hail mary')))
check('Fiction category excludes Nonfiction', lambda: (lambda d, b: (not d['categories'].fillna('').str.contains('nonfiction', case=False).any(), d['categories'].head(3).tolist()))(*search('world war 2', category_filter='Fiction')))
check('Science category excludes science fiction', lambda: (lambda d, b: (not d['categories'].fillna('').str.contains('science.?fiction', case=False).any(), d['categories'].head(3).tolist()))(*search('harry', category_filter='Science')))
check('"history of rome" + Computer Science -> clear "no match"', lambda: (lambda d, b: (len(d) == 0 and b.startswith('No Computer Science'), b))(*search('history of rome', category_filter='Computer Science')))
check('bad category in link is ignored, not a crash', lambda: (c.get('/?q=harry&category=Nope').status_code == 200, ''))
check('top page still works', lambda: (c.get('/top').status_code == 200, ''))

print(f"\n{sum(passed)}/{len(passed)} checks passed")
