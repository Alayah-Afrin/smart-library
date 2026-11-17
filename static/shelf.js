// Shelf + reading streak, stored only in this browser (localStorage). No login needed.
const Shelf = (() => {
  const KEY = 'sl_shelf_v1', STREAK = 'sl_streak_v1';
  const read = (k, fallback) => { try { return JSON.parse(localStorage.getItem(k)) ?? fallback; } catch { return fallback; } };
  const write = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: ignore */ } };

  const all = () => read(KEY, []);
  const has = key => all().some(b => b.key === key);
  function toggle(book) {
    let list = all();
    if (list.some(b => b.key === book.key)) list = list.filter(b => b.key !== book.key);
    else list.unshift({ ...book, added: Date.now(), progress: 0 });
    write(KEY, list);
    updateCount();
    return list.some(b => b.key === book.key);
  }
  function remove(key) { write(KEY, all().filter(b => b.key !== key)); updateCount(); }
  function setProgress(readerId, page, pages) {
    const list = all();
    const b = list.find(x => String(x.reader) === String(readerId));
    if (b) { b.progress = page; b.pages = pages; b.lastRead = Date.now(); write(KEY, list); }
  }
  function updateCount() {
    const el = document.getElementById('shelf-count');
    if (!el) return;
    const n = all().length;
    el.textContent = n;
    el.hidden = n === 0;
  }

  // streak: consecutive calendar days with a visit
  function tickStreak() {
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 864e5).toDateString();
    const s = read(STREAK, { last: '', days: 0 });
    if (s.last !== today) {
      s.days = s.last === yesterday ? s.days + 1 : 1;
      s.last = today;
      write(STREAK, s);
    }
    const el = document.getElementById('streak');
    if (el && s.days >= 2) { el.textContent = `🔥 ${s.days}-day streak`; el.hidden = false; }
  }

  // The "Search" tab takes you back to your last search; the "Smart Library" title starts fresh.
  function rememberSearch() {
    const LAST = 'sl_last_search';
    let last = '';
    try {
      if (location.pathname === '/') {
        const p = new URLSearchParams(location.search);
        if (p.get('q') || p.get('category') || p.get('free')) sessionStorage.setItem(LAST, location.search);
        else sessionStorage.removeItem(LAST);           // plain home page = a fresh start
      }
      last = sessionStorage.getItem(LAST) || '';
    } catch { /* storage blocked: tab just goes home */ }
    const tab = document.querySelector('.topnav a[href="/"]');
    if (tab && last) { tab.href = '/' + last; tab.title = 'Back to your last search'; }
  }

  document.addEventListener('DOMContentLoaded', () => { updateCount(); tickStreak(); rememberSearch(); });
  return { all, has, toggle, remove, setProgress };
})();

// wire every "Save" button on the page
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.save-btn').forEach(btn => {
    const d = btn.dataset;
    const icon = btn.classList.contains('icon-only');
    const paint = on => {
      btn.classList.toggle('saved', on);
      btn.textContent = icon ? (on ? '♥' : '♡') : (on ? '♥ Saved' : '♡ Save');
      btn.setAttribute('aria-pressed', on);
      btn.setAttribute('aria-label', on ? 'Saved — click to remove from my shelf' : 'Save to my shelf');
      btn.title = btn.getAttribute('aria-label');
    };
    paint(Shelf.has(d.key));
    btn.addEventListener('click', () => paint(Shelf.toggle({
      key: d.key, title: d.title, author: d.author, cover: d.cover, read: d.read, reader: d.reader || '', pdf: d.pdf || ''
    })));
  });
});

// Night / day mode button in the top menu (the choice is remembered in this browser)
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('theme-btn');
  if (!btn) return;
  const paint = () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = dark ? '☀️' : '🌙';
    const label = dark ? 'Switch to day mode' : 'Switch to night mode';
    btn.setAttribute('aria-label', label); btn.title = label;
  };
  paint();
  btn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('sl_theme', next); } catch (e) { /* private mode: just this page */ }
    paint();
  });
});
