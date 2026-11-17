// Community ratings + comments inside the summary flyout / drawer. Builds DOM safely (no innerHTML with user text).
const Community = (() => {
  const token = () => (document.querySelector('meta[name="csrf-token"]') || {}).content || '';
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  async function api(method, url, body) {
    const res = await fetch(url, {
      method, headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token() },
      body: body ? JSON.stringify(body) : undefined, credentials: 'same-origin'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || 'Something went wrong. Please try again.');
    return data;
  }

  function draw(box, key, s) {
    box.replaceChildren();
    box.className = 'community';
    const head = el('h4', '', 'Readers here');
    const avg = el('p', 'community-avg', s.count ? `${s.avg.toFixed(1)}★ from ${s.count} reader${s.count === 1 ? '' : 's'}` : 'No ratings yet — be the first.');
    box.append(head, avg);

    // what the comments feel like (sentiment model on the written comments)
    if (s.sentiment) {
      const f = s.sentiment, wrap = el('div', 'feel');
      wrap.appendChild(el('p', 'feel-verdict', `${f.verdict} · from ${f.total} comment${f.total === 1 ? '' : 's'}`));
      const bar = el('div', 'feel-bar'); bar.setAttribute('role', 'img');
      bar.setAttribute('aria-label', `${f.percent.positive}% positive, ${f.percent.neutral}% neutral, ${f.percent.negative}% negative`);
      ['positive', 'neutral', 'negative'].forEach(k => {
        if (!f.percent[k]) return;
        const seg = el('span', 'feel-' + k); seg.style.width = f.percent[k] + '%'; seg.title = `${f.percent[k]}% ${k}`;
        bar.appendChild(seg);
      });
      const legend = el('p', 'feel-legend', `😊 ${f.percent.positive}%   😐 ${f.percent.neutral}%   😞 ${f.percent.negative}%`);
      wrap.append(bar, legend);
      box.appendChild(wrap);
    }

    // rating form
    const form = el('form', 'community-form');
    const stars = el('div', 'star-pick'); stars.setAttribute('role', 'radiogroup'); stars.setAttribute('aria-label', 'Your rating');
    let chosen = s.mine ? s.mine.rating : 0;
    const paint = () => [...stars.children].forEach((b, i) => { b.classList.toggle('on', i < chosen); b.setAttribute('aria-checked', i + 1 === chosen); });
    for (let i = 1; i <= 5; i++) {
      const b = el('button', '', '★'); b.type = 'button'; b.setAttribute('role', 'radio'); b.title = `${i} star${i > 1 ? 's' : ''}`;
      b.onclick = () => { chosen = i; paint(); };
      stars.appendChild(b);
    }
    paint();
    const text = el('textarea'); text.maxLength = 500; text.rows = 2; text.placeholder = 'A short comment (optional)';
    text.value = s.mine ? s.mine.comment : '';
    const send = el('button', 'pager-btn', s.mine ? 'Update' : 'Post'); send.type = 'submit';
    const msg = el('p', 'community-msg');
    form.append(stars, text, send, msg);
    form.onsubmit = async e => {
      e.preventDefault();
      if (!chosen) { msg.textContent = 'Pick 1 to 5 stars first.'; return; }
      send.disabled = true;
      try { draw(box, key, await api('POST', '/api/community', { key, rating: chosen, comment: text.value })); }
      catch (err) { msg.textContent = err.message; send.disabled = false; }
    };
    box.appendChild(form);

    // comments
    const list = el('ul', 'comment-list');
    s.comments.forEach(c => {
      const li = el('li');
      li.append(el('span', 'comment-stars', '★'.repeat(c.rating)), el('span', 'comment-date', ' ' + c.created_at.slice(0, 10)));
      const face = { positive: '😊 positive', neutral: '😐 neutral', negative: '😞 negative' }[c.sentiment];
      if (face) li.appendChild(el('span', 'feel-tag feel-tag-' + c.sentiment, face));
      li.appendChild(el('p', '', c.comment));
      // Report: pick a reason first (spam / offensive / spoiler / other)
      const rep = el('button', 'report-btn', 'Report'); rep.type = 'button';
      const pick = el('div', 'report-pick'); pick.hidden = true;
      pick.appendChild(el('span', 'report-ask', 'Why?'));
      [['spam', 'Spam'], ['offensive', 'Offensive'], ['spoiler', 'Spoiler'], ['other', 'Other']].forEach(([key, label]) => {
        const b = el('button', 'report-reason', label); b.type = 'button';
        b.onclick = async () => {
          pick.querySelectorAll('button').forEach(x => { x.disabled = true; });
          try {
            const r = await api('POST', '/api/community/report/' + c.id, { reason: key });
            pick.replaceChildren(el('span', 'report-done', r.already ? 'You already reported this.' : 'Thanks, reported as ' + label.toLowerCase() + '.'));
          } catch (err) {
            pick.replaceChildren(el('span', 'report-done', err.message));
          }
        };
        pick.appendChild(b);
      });
      const cancel = el('button', 'report-reason report-cancel', 'Cancel'); cancel.type = 'button';
      cancel.onclick = () => { pick.hidden = true; rep.hidden = false; };
      pick.appendChild(cancel);
      rep.onclick = () => { rep.hidden = true; pick.hidden = false; };
      li.append(rep, pick);
      list.appendChild(li);
    });
    if (s.comments.length) box.appendChild(list);
  }

  async function render(box, key) {
    if (!box) return;
    box.replaceChildren();
    if (!key) return;                       // books found online only: nothing stored for them
    box.className = 'community';
    box.appendChild(el('p', 'community-avg', 'Loading reviews…'));
    try { draw(box, key, await api('GET', '/api/community?key=' + encodeURIComponent(key))); }
    catch { box.replaceChildren(el('p', 'community-avg', 'Reviews could not be loaded.')); }
  }
  return { render };
})();
