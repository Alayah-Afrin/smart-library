"""Sentiment analysis for community review comments.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa trained on ~124M tweets, fine-tuned for
3-class sentiment). Unlike binary SST-2 models it has a real NEUTRAL class, so "it was okay" isn't forced
into positive. Short, informal text (like tweets) is close to what review comments look like.
Low-confidence results are also treated as neutral.

The model loads in a background thread at startup, so the website never waits for it.
If it can't load (no internet on first run, missing package), reviews still save; their sentiment is
filled in later by analyse_missing() once the model is available.
"""
import threading

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
NEUTRAL_BELOW = 0.75          # positive/negative with confidence under this = neutral ("it was okay I guess" = 0.72)

_clf = None
_status = 'not loaded'
_lock = threading.Lock()
_run_lock = threading.Lock()


def _load(on_ready=None):
    global _clf, _status
    try:
        from transformers import pipeline
        _status = 'loading'
        clf = pipeline('sentiment-analysis', model=MODEL)
        clf('warm up')
        _clf, _status = clf, 'ready'
    except Exception as e:                       # offline, package missing, ...
        _status = f'unavailable ({type(e).__name__})'
        return
    if on_ready:
        try:
            on_ready()
        except Exception as e:
            print('sentiment backfill failed:', repr(e))


def start_loading(on_ready=None):
    """Begin loading the model without blocking the caller; run on_ready() once it is loaded."""
    global _status
    with _lock:
        if _status == 'not loaded' or _status.startswith('unavailable'):
            _status = 'loading'
            threading.Thread(target=_load, args=(on_ready,), daemon=True).start()


def status():
    return _status


def load_now():
    """Blocking load (used by scripts and tests)."""
    if _clf is None:
        _load()
    return _clf is not None


def analyse(texts):
    """List of strings -> list of (label, score) with label in positive / negative / neutral.
    Empty text -> (None, None). Returns None for every item if the model isn't ready."""
    if _clf is None:
        return [(None, None)] * len(texts)
    out = [(None, None)] * len(texts)
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not idx:
        return out
    with _run_lock:                              # the shared model isn't safe for two threads at once
        preds = _clf([texts[i][:1000] for i in idx], truncation=True)
    for i, p in zip(idx, preds):
        score = float(p['score'])
        label = 'neutral' if score < NEUTRAL_BELOW else p['label'].lower()
        out[i] = (label, round(score, 3))
    return out


def mismatch(rating, label):
    """Stars and words disagree, e.g. 5★ with a clearly negative comment (often spam or a mistake)."""
    if not label or label == 'neutral' or rating is None:
        return False
    return (rating >= 4 and label == 'negative') or (rating <= 2 and label == 'positive')
