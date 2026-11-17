from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort, flash, send_file
from functools import lru_cache
import datetime
import hmac
import os
import re
import secrets
import sys
sys.path.append('src')
from search import load_search_system, search_books, top_books, todays_pick, online_extra, CATEGORY_GROUPS, USER_AGENT
from descriptions import fetch_description, Fail
import community
import reader
import sentiment
import pdfs
import fetch_data

app = Flask(__name__)


def _secret_key():
    """SECRET_KEY env var, else a random key kept in instance/secret_key (so sessions survive restarts)."""
    if os.environ.get('SECRET_KEY'):
        return os.environ['SECRET_KEY']
    path = os.path.join('instance', 'secret_key')
    os.makedirs('instance', exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.secret_key = _secret_key()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')      # moderation page is off unless this is set

print("Loading search system... this may take a moment.")
fetch_data.ensure_data()                    # on the website: downloads the book data on first start
system = load_search_system(fetch_data.DATA_DIR)
community.init()


def backfill_sentiment():
    """Analyse comments that were saved before the sentiment model finished loading."""
    rows = community.missing_sentiment()
    if rows:
        labels = sentiment.analyse([r['comment'] for r in rows])
        community.set_sentiment([(r['id'], lab, sc) for r, (lab, sc) in zip(rows, labels)])
        print(f"Sentiment added to {len(rows)} older comments.")


# review sentiment model loads in the background (set SENTIMENT=0 to switch it off)
if os.environ.get('SENTIMENT', '1') != '0':
    sentiment.start_loading(on_ready=backfill_sentiment)

TITLE_INDEX = [(t.lower(), t) for t in system['title_list']]
AUTHOR_INDEX = [(a.lower(), a) for a in system['author_list']]
KNOWN_NAMES = set(system['title_list']) | set(system['author_list'])   # a banner that is one of these = "did you mean"
MOODS = ['joy', 'sadness', 'fear', 'surprise', 'anger', 'disgust']
OL_WORK = re.compile(r'^/works/OL\d{1,12}W$')          # Open Library work id, e.g. /works/OL82563W
# Online (Open Library) books that really appeared in a search. Only these can be rated, so nobody can
# script ratings for made-up ids. Kept small: oldest ids are forgotten first.
SHOWN_ONLINE = {}
SHOWN_ONLINE_MAX = 50_000

# Gutenberg books we can open in our own reader: gutenberg id -> (title, author)
_wd = system['work_dedup']
READABLE = {}
for t, a, u in zip(_wd['title'], _wd['author'], _wd['read_url']):
    gid = reader.gutenberg_id(u)
    if gid and gid not in READABLE:
        READABLE[gid] = (t, a)
PDF_BOOKS, PDF_ID_BY_URL = pdfs.load_list()     # free textbooks whose PDF we can show in our reader
print(f"Books readable on this site: {len(READABLE)} texts + {len(PDF_BOOKS)} textbook PDFs")
print("Ready.")


# ---------- small helpers ----------

def decorate(records):
    """Add the in-site reader id and the community rating to each result card."""
    for r in records:
        # one id for reviews + shelf: our catalog key, or Open Library's work id for books found online
        dk, ol = r.get('dedup_key'), r.get('ol_key')
        r['book_key'] = dk if isinstance(dk, str) and dk else (f'ol:{ol}' if isinstance(ol, str) and OL_WORK.match(ol) else '')
        if r['book_key'].startswith('ol:'):
            SHOWN_ONLINE[r['book_key']] = True
            if len(SHOWN_ONLINE) > SHOWN_ONLINE_MAX:
                SHOWN_ONLINE.pop(next(iter(SHOWN_ONLINE)))
    stats = community.averages([r['book_key'] for r in records])
    for r in records:
        why = [w for w in str(r.get('why') or '').split('|') if w and w != 'Free to read' and '★' not in w]
        r['reason'] = why[0] if why else ''           # one short "why this book" label for the card
        url = r.get('read_url') if isinstance(r.get('read_url'), str) else ''
        gid = reader.gutenberg_id(url)
        r['reader_id'] = gid if gid in READABLE else None
        r['pdf_id'] = PDF_ID_BY_URL.get(url)
        avg, n = stats.get(r['book_key'], (None, 0))
        r['community_avg'], r['community_n'] = avg, n
    return records


def csrf_token():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_urlsafe(24)
    return session['csrf']


app.jinja_env.globals['csrf_token'] = csrf_token


@app.template_filter('short_cats')
def short_cats(text, n=2):
    """'Textbook; Humanities; Philosophy' -> 'Textbook · Humanities' (full list stays in the tooltip)."""
    parts = []
    for p in re.split(r'[;,|]', str(text or '')):
        p = p.replace('Category:', '').strip()
        if p and p.lower() not in {x.lower() for x in parts}:
            parts.append(p)
    return ' · '.join(parts[:n])


def check_csrf():
    sent = request.form.get('csrf', '') or request.headers.get('X-CSRF-Token', '')
    if not sent or not hmac.compare_digest(sent, session.get('csrf', '')):
        abort(400, description='Invalid or missing security token. Reload the page and try again.')


@app.before_request
def ensure_anon_id():
    # anonymous id used only to allow one rating per book per browser (not an identity)
    if 'anon' not in session:
        session['anon'] = secrets.token_urlsafe(16)
        session.permanent = True


app.permanent_session_lifetime = datetime.timedelta(days=365)


# ---------- search pages ----------

def pick_matches(index, q, limit):
    starts, contains, seen = [], [], set()
    for low, original in index:
        if low in seen:
            continue
        if low.startswith(q):
            starts.append(original)
            seen.add(low)
        elif q in low and len(contains) < limit:
            contains.append(original)
            seen.add(low)
        if len(starts) >= limit:
            break
    return (starts + contains)[:limit]


@app.route('/suggest')
def suggest():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify([])
    items = [{'text': t, 'type': 'Title'} for t in pick_matches(TITLE_INDEX, q, 5)]
    items += [{'text': a, 'type': 'Author'} for a in pick_matches(AUTHOR_INDEX, q, 3)]
    return jsonify(items)


PICK_CACHE = {}


def get_pick():
    today = datetime.date.today()
    if PICK_CACHE.get('day') != today:
        p = todays_pick(system)
        desc = p.get('description') if p else ''
        if p and isinstance(desc, str) and desc.startswith('Subjects:'):
            # Gutenberg books keep their subject list here; that is not a summary
            p['subjects'] = desc[len('Subjects:'):].strip()
            p['description'] = ''
        if p and not (isinstance(p.get('description'), str) and len(p['description']) >= 40):
            try:
                isbn = p.get('isbn13') if isinstance(p.get('isbn13'), str) else ''
                p['description'] = fetch_description(p['title'], p['author'], isbn)
            except Exception:
                p['description'] = ''
        if p:
            decorate([p])
        PICK_CACHE['day'] = today
        PICK_CACHE['pick'] = p
    return PICK_CACHE['pick']


@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    suggestion = None
    query = ''
    free_only = False
    mood_filter = category_filter = None
    did_you_mean = None
    online_q = ''
    # Searches are normal links (/?q=harry&category=Fiction&free=1) so Back, refresh and sharing work.
    # POST with 'query' is still accepted for older links/forms.
    v = request.values
    searching = 'q' in v or 'query' in v or 'category' in v or 'free' in v
    pick = None if searching else get_pick()

    if searching:
        query = (v.get('q') if v.get('q') is not None else v.get('query', ''))[:300]
        mood_filter = v.get('mood') or None
        if mood_filter not in MOODS:
            mood_filter = None
        category_filter = v.get('category') or None
        if category_filter not in CATEGORY_GROUPS:
            category_filter = None
        free_only = v.get('free') == '1'

        results_df, suggestion = search_books(
            query, system,
            mood_filter=mood_filter,
            category_filter=category_filter,
            top_k=24,
            free_only=free_only,
            online=False,                 # the page fetches "Also found online" afterwards (/online)
        )
        words = query.lower().split()
        if (query.strip() and len(words) <= 4 and not category_filter and not free_only and not mood_filter
                and not set(words) & {'by', 'author', 'writer', 'written'}):
            online_q = query.strip()
        alternatives = results_df.attrs.get('alternatives') or []
        results = decorate(results_df.fillna('').to_dict('records'))
        if alternatives:
            did_you_mean = alternatives               # several meanings: every one is a link
        elif suggestion and suggestion.startswith('Did you mean: '):
            did_you_mean, suggestion = suggestion[len('Did you mean: '):], None
        elif suggestion in KNOWN_NAMES and suggestion.lower() != query.strip().lower():
            did_you_mean, suggestion = suggestion, None
    return render_template('index.html', pick=pick, results=results, suggestion=suggestion,
                           did_you_mean=did_you_mean, query=query, online_q=online_q,
                           category_groups=list(CATEGORY_GROUPS.keys()), selected_category=category_filter,
                           moods=MOODS, selected_mood=mood_filter, free_checked=free_only)


@app.route('/top')
def top():
    category = request.args.get('category') or None
    results_df, banner = top_books(system, category_filter=category, top_k=24)
    results = decorate(results_df.fillna('').to_dict('records'))
    return render_template('index.html', results=results, suggestion=banner, query='', page='top',
                           category_groups=list(CATEGORY_GROUPS.keys()),
                           selected_category=category or '')


@app.route('/shelf')
def shelf():
    # the shelf lives in the visitor's browser (localStorage); the page fills itself in
    return render_template('shelf.html', page='shelf')


DESC_CACHE = {}


@app.route('/describe')
def describe():
    title = request.args.get('title', '')[:200]
    author = request.args.get('author', '')[:100]
    isbn = request.args.get('isbn', '')[:20]
    key = (title, author)
    if key in DESC_CACHE:
        return jsonify(text=DESC_CACHE[key])
    try:
        text = fetch_description(title, '' if author == 'Unknown author' else author, isbn)
    except Fail:
        return jsonify(text='', error=True)
    DESC_CACHE[key] = text
    return jsonify(text=text)


@app.route('/online')
def online():
    """Late "Also found online" section for the results page (HTML snippet, empty if nothing new)."""
    q = request.args.get('q', '').strip()[:300]
    df = online_extra(q, system) if q else None
    if df is None or len(df) == 0:
        return ''
    return render_template('_online.html', books=decorate(df.fillna('').to_dict('records')))


# ---------- in-site reader ----------

@lru_cache(maxsize=32)
def _book_pages(gid):
    return reader.load_book(gid, USER_AGENT)


@app.route('/read/<int:gid>')
def read(gid):
    if gid not in READABLE:                    # only books from our own catalog
        abort(404)
    title, author = READABLE[gid]
    source = f'https://www.gutenberg.org/ebooks/{gid}'
    try:
        pages = _book_pages(gid)
    except reader.ReaderError:
        return render_template('reader.html', page='read', title=title, author=author, gid=gid,
                               source=source, error=True), 503
    n = len(pages)
    p = min(max(request.args.get('page', 1, type=int), 1), n)
    return render_template('reader.html', page='read', title=title, author=author, gid=gid, source=source,
                           paras=pages[p - 1], page_no=p, pages=n, error=False)


# ---------- textbook PDFs inside our reader ----------

@app.route('/read/pdf/<int:pid>')
def read_pdf(pid):
    if not 0 <= pid < len(PDF_BOOKS):
        abort(404)
    b = PDF_BOOKS[pid]
    return render_template('reader_pdf.html', page='read', pid=pid, title=b['title'],
                           source=b['read_url'], pdf_url=b['pdf_url'])


@app.route('/pdf/<int:pid>')
def pdf_file(pid):
    if not 0 <= pid < len(PDF_BOOKS):
        abort(404)
    b = PDF_BOOKS[pid]
    st = pdfs.status(b['pdf_url'], USER_AGENT)               # starts the background download if needed
    if st['state'] == 'error':
        return render_template('pdf_unavailable.html', title=b['title'], pdf_url=b['pdf_url'],
                               source=b['read_url']), 502
    if st['state'] != 'ready':
        return render_template('pdf_unavailable.html', title=b['title'], pdf_url=b['pdf_url'],
                               source=b['read_url'], preparing=st), 202
    return send_file(os.path.abspath(pdfs.cached_path(b['pdf_url'])), mimetype='application/pdf',
                     conditional=True, download_name=f"{b['title'][:80]}.pdf".replace('/', '-'))


@app.route('/pdf/<int:pid>/status')
def pdf_status(pid):
    if not 0 <= pid < len(PDF_BOOKS):
        abort(404)
    return jsonify(pdfs.status(PDF_BOOKS[pid]['pdf_url'], USER_AGENT))


# ---------- community ratings + comments ----------

def _book_key():
    key = (request.values.get('key') or (request.get_json(silent=True) or {}).get('key') or '').strip()
    online_ok = key.startswith('ol:') and OL_WORK.match(key[3:]) and (key in SHOWN_ONLINE or community.has_reviews(key))
    known = key in system['dedup_keys'] or online_ok
    if not key or len(key) > 400 or not known:
        abort(400, description='Unknown book.')
    return key


@app.get('/api/community')
def community_get():
    return jsonify(community.summary(_book_key(), session.get('anon', '')))


@app.post('/api/community')
def community_post():
    check_csrf()
    data = request.get_json(silent=True) or {}
    key = _book_key()
    try:
        rating = int(data.get('rating', 0))
    except (TypeError, ValueError):
        rating = 0
    if not 1 <= rating <= 5:
        return jsonify(error='Choose 1 to 5 stars.'), 400
    comment = ' '.join(str(data.get('comment', '')).split())[:500]
    label, score = sentiment.analyse([comment])[0]       # (None, None) if no comment or model still loading
    community.save(key, session['anon'], rating, comment, label, score)
    out = community.summary(key, session['anon'])
    out['my_sentiment'] = label
    out['stars_words_disagree'] = sentiment.mismatch(rating, label)
    return jsonify(out)


@app.post('/api/community/report/<int:review_id>')
def community_report(review_id):
    check_csrf()
    reason = str((request.get_json(silent=True) or {}).get('reason', '')).lower()
    if reason not in community.REPORT_REASONS:
        return jsonify(error='Please choose a reason: ' + ', '.join(community.REPORT_REASONS.values()) + '.'), 400
    new = community.report(review_id, session['anon'], reason)      # one report per comment per browser
    return jsonify(ok=True, already=not new)


# ---------- moderation (only when ADMIN_PASSWORD is set) ----------

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not ADMIN_PASSWORD:
        abort(404)
    if request.method == 'POST':
        check_csrf()
        if request.form.get('password') is not None:
            if hmac.compare_digest(request.form['password'].encode('utf-8'), ADMIN_PASSWORD.encode('utf-8')):
                session['admin'] = True
            else:
                flash('Wrong password.')
        elif session.get('admin'):
            rid = request.form.get('id', type=int)
            if request.form.get('action') == 'delete':
                community.delete(rid)
            elif request.form.get('action') == 'keep':
                community.clear_reports(rid)
        return redirect(url_for('admin'))
    is_admin = session.get('admin', False)
    return render_template('admin.html', page='admin', logged_in=is_admin,
                           reviews=community.needs_review() if is_admin else [],
                           site=community.site_sentiment() if is_admin else None,
                           model_status=sentiment.status())


if __name__ == '__main__':
    # debug mode only when asked for:  set FLASK_DEBUG=1  (never on a server other people can reach)
    # PORT lets you run next to another copy, e.g.  $env:PORT = "5057"
    app.run(port=int(os.environ.get('PORT', '5000')), debug=os.environ.get('FLASK_DEBUG') == '1')
