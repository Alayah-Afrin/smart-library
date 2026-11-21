"""Learning-path planner: "I'm a 3rd-year student and want to learn ML from scratch" -> a route of steps,
each with why it comes first, about how long it takes, and 2-3 real books from our catalog.

The subject map below is hand-written (prerequisites in order). Subjects not in the map get a generic
introduction -> core -> advanced route built from the search engine.
"""
import re

from rapidfuzz import fuzz

from search import fold, search_books, STOPWORDS


def step(label, query, weeks, why, foundation=False, words=None):
    """One step of a route. `words`: what the student might say they already know ("i know python")."""
    return {'label': label, 'query': query, 'weeks': weeks, 'why': why, 'foundation': foundation,
            'words': words or [w for w in fold(label).replace('&', ' ').split() if len(w) > 3]}


PROGRAMMING = step('Programming basics with Python', 'python programming', 3,
                   'Almost every computing subject is learned by writing code; Python is the easiest start.',
                   foundation=True, words=['python', 'programming', 'coding'])
ALGEBRA = step('Algebra & functions', 'college algebra', 2,
               'Equations, functions and graphs are the language the later maths is written in.',
               foundation=True, words=['algebra'])
CALCULUS = step('Calculus', 'calculus', 3,
                'Rates of change and optimisation: how things move, grow, and how models "learn".',
                foundation=True, words=['calculus'])
LINEAR_ALGEBRA = step('Linear algebra', 'linear algebra', 3,
                      'Vectors and matrices: how data, images and models are stored and transformed.',
                      foundation=True, words=['linear algebra', 'matrices', 'matrix'])
PROBABILITY = step('Probability & statistics', 'probability and statistics', 3,
                   'Uncertainty, distributions and testing: how to tell a real pattern from luck.',
                   foundation=True, words=['probability', 'statistics', 'stats'])
DISCRETE = step('Discrete mathematics', 'discrete mathematics', 2,
                'Logic, sets and graphs: the maths behind algorithms and computer science.',
                foundation=True, words=['discrete'])
DSA = step('Data structures & algorithms', 'data structures and algorithms', 4,
           'Organising data and solving problems efficiently: the core of computer science.',
           words=['data structures', 'algorithms', 'dsa'])
ML = step('Machine learning foundations', 'machine learning', 5,
          'The core methods: regression, classification, trees, clustering, and how to evaluate them.',
          words=['machine learning'])
DL = step('Deep learning & neural networks', 'deep learning neural networks', 4,
          'Neural networks for images, text and speech, built on everything before.',
          words=['deep learning', 'neural networks'])
DATA_SCIENCE = step('Data science in practice', 'data science', 2,
                    'Cleaning real data and carrying a project from question to result.',
                    words=['data science', 'pandas'])
DATABASES = step('Databases & SQL', 'database', 3, 'Storing and querying data reliably.',
                 words=['database', 'databases', 'sql'])
NETWORKS = step('Computer networks', 'computer networks', 3, 'How computers talk: the internet, protocols, security.',
                words=['networks', 'networking'])
OS = step('Operating systems', 'operating systems', 3, 'Processes, memory and files: what runs underneath every program.',
          words=['operating systems'])
ARCH = step('Computer organisation & architecture', 'computer architecture', 3,
            'How a processor actually executes your code.', foundation=True, words=['architecture'])
WEB = step('Web development (HTML, CSS, JavaScript)', 'web development html css javascript', 4,
           'Building pages and apps that run in a browser.', words=['html', 'css', 'javascript', 'web'])
SOFTWARE = step('Software engineering', 'software engineering', 3,
                'Working on real projects: design, testing, version control, teamwork.', words=['software engineering'])
AI_STEP = step('Artificial intelligence', 'artificial intelligence', 4,
               'Search, reasoning and planning: the classic ideas of AI, beside machine learning.',
               words=['artificial intelligence'])
PHYSICS = step('Physics', 'college physics', 4, 'Motion, energy, electricity and waves.', foundation=True,
               words=['physics'])
CIRCUITS = step('Circuit analysis', 'circuit analysis', 3, 'Voltage, current and components: the base of electronics.',
                words=['circuit', 'circuits'])
DIGITAL = step('Digital logic & electronics', 'digital logic electronics', 3,
               'Gates, flip-flops and how digital hardware works.', words=['digital', 'electronics'])
EMBEDDED = step('Embedded systems', 'embedded systems microcontrollers', 3,
                'Programming small computers that control real devices.', words=['embedded', 'microcontroller'])
CHEMISTRY = step('General chemistry', 'chemistry', 4, 'Atoms, bonds and reactions.', foundation=True,
                 words=['chemistry'])
ORGANIC = step('Organic chemistry', 'organic chemistry', 4, 'Carbon compounds: the chemistry of life and materials.',
               words=['organic'])
MICRO = step('Microeconomics', 'principles of microeconomics', 3, 'How people and firms decide: prices, markets.',
             words=['microeconomics'])
MACRO = step('Macroeconomics', 'principles of macroeconomics', 3, 'Whole economies: growth, inflation, money.',
             words=['macroeconomics'])
ACCOUNTING = step('Financial accounting', 'principles of accounting', 4, 'Recording and reading a business\'s numbers.',
                  words=['accounting'])

SUBJECTS = {
    'machine learning': (['machine learning', 'ml'], [PROGRAMMING, LINEAR_ALGEBRA, CALCULUS, PROBABILITY, ML, DL, DATA_SCIENCE]),
    'deep learning': (['deep learning', 'neural network', 'neural networks', 'dl'],
                      [PROGRAMMING, LINEAR_ALGEBRA, CALCULUS, PROBABILITY, ML, DL]),
    'data science': (['data science', 'data scientist', 'data analysis', 'data analytics'],
                     [PROGRAMMING, PROBABILITY, DATABASES, DATA_SCIENCE, ML]),
    'artificial intelligence': (['artificial intelligence', 'ai'],
                                [PROGRAMMING, DISCRETE, DSA, PROBABILITY, AI_STEP, ML]),
    'programming': (['programming', 'coding', 'python', 'learn to code'], [PROGRAMMING, DSA, SOFTWARE]),
    'web development': (['web development', 'web developer', 'website', 'websites', 'frontend', 'backend', 'full stack'],
                        [PROGRAMMING, WEB, DATABASES, NETWORKS]),
    'data structures & algorithms': (['data structures', 'algorithms', 'dsa', 'competitive programming'],
                                     [PROGRAMMING, DISCRETE, DSA]),
    'databases': (['database', 'databases', 'dbms', 'sql'], [PROGRAMMING, DISCRETE, DATABASES]),
    'computer networks': (['computer networks', 'networking', 'networks'], [PROGRAMMING, OS, NETWORKS]),
    'operating systems': (['operating systems', 'operating system', 'os'], [PROGRAMMING, ARCH, OS]),
    'calculus': (['calculus'], [ALGEBRA, CALCULUS]),
    'linear algebra': (['linear algebra', 'matrices'], [ALGEBRA, LINEAR_ALGEBRA]),
    'statistics': (['statistics', 'stats', 'probability'], [ALGEBRA, PROBABILITY]),
    'physics': (['physics'], [ALGEBRA, CALCULUS, PHYSICS]),
    'chemistry': (['chemistry'], [ALGEBRA, CHEMISTRY, ORGANIC]),
    'electronics': (['electronics', 'eee', 'electrical', 'circuits', 'embedded', 'embedded systems', 'microcontroller'],
                    [PHYSICS, CIRCUITS, DIGITAL, EMBEDDED]),
    'economics': (['economics', 'economy'], [ALGEBRA, MICRO, MACRO]),
    'accounting': (['accounting', 'bookkeeping'], [ALGEBRA, ACCOUNTING]),
}

FILLER = STOPWORDS | {'learn', 'learning', 'study', 'studying', 'understand', 'master', 'get', 'into', 'started', 'start',
                      'scratch', 'from', 'beginner', 'beginners', 'basics', 'basic', 'how', 'year', 'student', 'im', "i'm",
                      'am', 'my', 'in', 'of', 'and', 'with', 'zero', 'complete', 'become', 'good', 'at', 'is', 'some',
                      'know', 'already', 'familiar', 'what', 'should', 'need', 'help', 'path', 'roadmap', 'route', 'plan',
                      'within', 'month', 'months', 'week', 'weeks', 'day', 'days', 'years', 'time', 'same', 'make', 'want',
                      'way', 'quickly', 'fast'}
ORDINAL_YEAR = re.compile(r'\b(1st|first|2nd|second|3rd|third|4th|fourth|final|last)\s+year\b|\b(senior|junior|sophomore)\b')
FROM_SCRATCH = re.compile(r'\bfrom (scratch|zero|the beginning)\b|\b(beginner|new to|no experience|never)\b')
KNOWS = re.compile(r"\b(?:i|i'm|im|already)\s+(?:already\s+)?(?:know|knows|learned|learnt|studied|am familiar with|familiar with|"
                   r"good at|comfortable with)\s+([a-z0-9 ,&+-]+?)(?:\.|;|$|\bbut\b|\band want\b|\band i\b)")


def find_subject(goal_f):
    """-> subject key or None. Exact alias first (whole words), then a close spelling ('machin learning')."""
    best = None
    for key, (aliases, _) in SUBJECTS.items():
        for a in aliases:
            if re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', goal_f):
                if best is None or len(a) > best[1]:          # the longest alias wins ("deep learning" over "learning")
                    best = (key, len(a))
    if best:
        return best[0]
    for key, (aliases, _) in SUBJECTS.items():
        for a in aliases:
            if len(a) >= 6 and fuzz.partial_ratio(a, goal_f) >= 90:
                return key
    return None


def find_subjects(goal_f):
    """-> every subject named in the goal, in the order they appear ("python and ml" -> programming, machine learning).
    A longer alias claims its words first, so "neural networks" is deep learning, not computer networks."""
    hits = []
    for key, (aliases, _) in SUBJECTS.items():
        for a in aliases:
            for m in re.finditer(r'(?<!\w)' + re.escape(a) + r'(?!\w)', goal_f):
                hits.append((len(a), m.start(), m.end(), key))
    taken, found = [], []
    for length, start, end, key in sorted(hits, key=lambda h: -h[0]):
        if any(start < e and end > s for s, e in taken):
            continue
        taken.append((start, end))
        found.append((start, key))
    for m in re.finditer(r'[a-z0-9+#]+', goal_f):               # a misspelt one-word alias ("pythoin", "databse")
        if any(m.start() < e and m.end() > s for s, e in taken) or len(m.group()) < 5:
            continue
        for key, (aliases, _) in SUBJECTS.items():
            if any(' ' not in a and len(a) >= 5 and fuzz.ratio(a, m.group()) >= 85 for a in aliases):
                found.append((m.start(), key))
                break
    keys = []
    for _, key in sorted(found):
        if key not in keys:
            keys.append(key)
    return keys


NUMBER_WORDS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10, 'twelve': 12}
TIME_LIMIT = re.compile(r'\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve)[\s-]*'
                        r'(day|days|week|weeks|month|months|year|years)\b')


def time_budget(goal_f):
    """"within 1 month" / "in 6 weeks" / "2 months" -> weeks, or None when no time limit is given."""
    m = TIME_LIMIT.search(goal_f)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
    unit = m.group(2)
    weeks = {'day': n / 7, 'week': n, 'month': n * 4, 'year': n * 52}[unit.rstrip('s')]
    return max(1, round(weeks)) if n else None


def is_core(step_, keys):
    """The step that IS the subject (machine learning -> 'Machine learning foundations'), not a prerequisite."""
    label = fold(step_['label'])
    for key in keys:
        for a in SUBJECTS[key][0]:
            if a in step_['words'] or re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', label):
                return True
    return False


def fit_to_budget(steps, keys, budget):
    """The full route is longer than the time the student has: keep programming and the subject itself, mark the
    rest 'later' (maths is picked up as it is needed), and share the weeks out. Returns the weeks used."""
    todo = [s for s in steps if s['status'] != 'known']
    if not todo:
        return 0
    protected = [s for s in todo if s['label'] == PROGRAMMING['label'] or is_core(s, keys)]
    keep_n = max(len(protected), max(1, budget // 2))
    droppable = [s for s in todo if s not in protected]
    droppable.sort(key=lambda s: (not s['foundation'], -todo.index(s)))       # maths first, then the latest steps
    while len(todo) > keep_n and droppable:
        gone = droppable.pop(0)
        gone['status'] = 'later'
        todo.remove(gone)
    base = {id(s): (s['weeks'] if s['status'] == 'learn' else max(1, s['weeks'] // 2)) for s in todo}
    scale = budget / sum(base.values())
    plan_weeks = {id(s): max(1, round(base[id(s)] * scale)) for s in todo}
    while sum(plan_weeks.values()) > budget and max(plan_weeks.values()) > 1:
        biggest = max(todo, key=lambda s: plan_weeks[id(s)])
        plan_weeks[id(biggest)] -= 1
    while sum(plan_weeks.values()) < budget:
        biggest = max(todo, key=lambda s: base[id(s)])
        plan_weeks[id(biggest)] += 1
    for s in todo:
        s['weeks'], s['status'] = plan_weeks[id(s)], 'learn'
    return sum(plan_weeks.values())


def read_background(goal_f):
    """What the goal says about the student: their year, 'from scratch', and what they already know."""
    year = ORDINAL_YEAR.search(goal_f)
    year_word = (year.group(1) or year.group(2)) if year else ''
    advanced_student = year_word in ('2nd', 'second', '3rd', 'third', '4th', 'fourth', 'final', 'last', 'senior', 'junior')
    known = []
    for m in KNOWS.finditer(goal_f):
        known += [w.strip() for w in re.split(r',|\band\b|&', m.group(1)) if w.strip()]
    return {'year': year_word, 'advanced_student': advanced_student,
            'from_scratch': bool(FROM_SCRATCH.search(goal_f)), 'known': known}


def topic_words(goal_f):
    return [w for w in re.findall(r"[a-z0-9+#]+", goal_f) if w not in FILLER and not w.isdigit() and len(w) > 1]


def generic_route(goal_f):
    """Subject not in the map: introduction -> core -> deeper, built from the words of the goal."""
    words = topic_words(goal_f)
    if not words:
        return None, []
    topic = ' '.join(words[:4])
    return topic, [
        step(f'Introduction to {topic}', f'introduction to {topic}', 3, 'Start with a gentle, complete overview.'),
        step(f'{topic.title()}: core ideas', topic, 4, 'The main textbook-level material.'),
        step(f'Going further in {topic}', f'advanced {topic}', 4, 'Deeper and more specialised once the basics are solid.'),
    ]


GENERIC = {'introduction', 'intro', 'advanced', 'principles', 'college', 'fundamentals', 'basics', 'core', 'ideas',
           'going', 'further', 'guide', 'practice'}


def fits(keys, need, title, categories):
    """How well a book suits a step: 2 = clearly (the words are in its TITLE), 1 = probably (the words form a
    subject phrase in one of its categories, e.g. "Dive into Deep Learning" filed under "Machine Learning";
    or, for our checked textbooks, a whole category like "Python"), 0 = no.
    "Data Structures" filed under "Computer Science" is NOT a data-science book."""
    title = fold(re.sub(r'\([^)]*\)', ' ', str(title)))    # "(McGraw-Hill Computer Science Series)" is not the topic
    cats = [fold(c).strip() for c in str(categories).split(';') if c.strip()]
    phrase = re.compile(r'(?<!\w)' + r'\W+'.join(map(re.escape, keys))) if keys else None
    in_title = sum(1 for w in keys if re.search(r'(?<!\w)' + re.escape(w), title))
    if phrase and phrase.search(title) or (keys and in_title >= max(need, len(keys) if len(keys) <= 2 else need)):
        return 2
    if len(keys) >= 2 and any(phrase.search(c) for c in cats):
        return 1
    if 'textbook' in cats and in_title + sum(1 for w in keys if w in cats) >= need:
        return 1
    return 0


_CANDIDATES = {}           # step query -> suitable books, best first (the same for everyone, so kept)


def candidates(sys, query):
    """Suitable books for one step, best first: books clearly about it (free ones first), then books that are
    probably about it. Free books are searched on their own so older popular books can't push them out."""
    if query in _CANDIDATES:
        return _CANDIDATES[query]
    keys = [w for w in fold(query).split() if w not in FILLER and w not in GENERIC and len(w) > 2]
    need = len(keys) if len(keys) <= 2 else 2
    groups = {(2, True): [], (2, False): [], (1, True): [], (1, False): []}
    seen = set()
    for free_only in (True, False):
        df, _ = search_books(query, sys, top_k=40, free_only=free_only, online=False)
        if df is None or len(df) == 0:
            continue
        df = df.fillna('')
        if not free_only:
            df = df.sort_values('n_ratings', ascending=False, kind='stable')
        for _, r in df.iterrows():
            k = r.get('dedup_key') or r.get('title')
            if not k or k in seen:
                continue
            score = fits(keys, need, r.get('title', ''), r.get('categories', ''))
            if score:
                seen.add(k)
                groups[(score, free_only)].append(r.to_dict())
    found = [b for g in groups.values() for b in g][:12]
    _CANDIDATES[query] = found
    return found


def warm_up(sys):
    """Prepare every mapped step once (run in the background at startup) so routes appear instantly."""
    for q in dict.fromkeys(st['query'] for _, steps in SUBJECTS.values() for st in steps):
        try:
            candidates(sys, q)
        except Exception as e:                      # never let warm-up break the app
            print('learning-path warm-up failed for', q, repr(e))


def pick_books(sys, query, used, n=3):
    """2-3 books for a step, never one already used by an earlier step of the same route."""
    picked = []
    for r in candidates(sys, query):
        k = r.get('dedup_key') or r.get('title')
        if k in used:
            continue
        picked.append(r)
        used.add(k)
        if len(picked) == n:
            break
    return picked


def plan(goal, sys):
    """-> dict with subject, background, steps (each with books), total weeks; or {'error': ...}."""
    goal = (goal or '').strip()[:300]
    if not goal:
        return {'error': 'Tell us what you want to learn, e.g. "I\'m a 3rd-year student and want to learn ML from scratch".'}
    goal_f = fold(goal)
    background = read_background(goal_f)
    budget = time_budget(goal_f)
    keys = find_subjects(goal_f)
    if not keys and find_subject(goal_f):             # a close spelling ("machin learning")
        keys = [find_subject(goal_f)]
    key = keys[0] if keys else None
    if key:
        # several subjects ("python and ml"): the longest route first, then the extra steps the others add
        known_now = ' '.join(background['known'])                # "i already know python": not something to learn
        learning = [k for k in keys if not any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', known_now)
                                               for a in SUBJECTS[k][0])]
        keys = learning or keys
        label = ' + '.join(keys)
        steps, seen_labels = [], set()
        for k in sorted(keys, key=lambda k: -len(SUBJECTS[k][1])):
            for s in SUBJECTS[k][1]:
                if s['label'] not in seen_labels:
                    seen_labels.add(s['label'])
                    steps.append(dict(s))
    else:
        label, steps = generic_route(goal_f)
        if not steps or not any(w in sys['vocab'] for w in topic_words(goal_f)):
            return {'error': 'We couldn\'t tell what you want to learn. Try naming the subject, e.g. "statistics" or "web development".'}
        steps = [dict(s) for s in steps]

    known_text = ' '.join(background['known'])
    used, total = set(), 0
    for s in steps:
        s['status'] = 'learn'
        if known_text and any(w in known_text for w in s['words']):
            s['status'] = 'known'                        # "i already know python"
        elif s['foundation'] and background['advanced_student']:
            s['status'] = 'review'                       # 3rd-year student: probably seen it, quick review
        if s['status'] != 'known':
            total += s['weeks'] if s['status'] == 'learn' else max(1, s['weeks'] // 2)
    full_weeks, fast_track = total, False
    if key and budget and total > budget:
        total, fast_track = fit_to_budget(steps, keys, budget), True
    for s in steps:
        s['books'] = pick_books(sys, s['query'], used) if s['status'] != 'later' else []
    if not key:
        steps = [s for s in steps if s['books']]          # a generic step with no matching book is dropped
        if not steps:
            return {'error': f'We don\'t have books for "{label}" yet. Try a related subject or a simpler name.'}
    return {'subject': label, 'goal': goal, 'background': background, 'steps': steps, 'weeks': total,
            'mapped': key is not None, 'budget': budget, 'fast_track': fast_track, 'full_weeks': full_weeks}
