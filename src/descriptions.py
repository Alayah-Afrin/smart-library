"""Look up one book's description on Open Library (free public API). Used by the /describe route."""
import re
import requests

OL = 'https://openlibrary.org'
HEAD = {'User-Agent': 'SmartLibrary-student-project'}


class Fail(Exception):
    """Open Library did not answer properly (busy or offline). Not the same as 'no description'."""


def get_json(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEAD, timeout=6)
    except Exception:
        raise Fail()
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        return None
    raise Fail()


def clean(text):
    text = re.sub(r'\[\d+\]|\(\[source\][^)]*\)', '', text)
    text = re.sub(r'\s*-{3,}.*$', '', text, flags=re.S)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 40 else ''


def _text(d):
    d = d.get('value') if isinstance(d, dict) else d
    return clean(d) if isinstance(d, str) else ''


def fetch_description(title, author, isbn13=''):
    work_key = None
    if isbn13:
        ed = get_json(f'{OL}/isbn/{isbn13}.json')
        if ed and ed.get('works'):
            work_key = ed['works'][0]['key']
            t = _text(ed.get('description'))
            if t:
                return t
    if not work_key:
        s = get_json(f'{OL}/search.json', {'title': title, 'author': author, 'limit': 1, 'fields': 'key'})
        if s and s.get('docs'):
            work_key = s['docs'][0]['key']
    if not work_key:
        return ''
    w = get_json(f'{OL}{work_key}.json')
    return _text((w or {}).get('description'))