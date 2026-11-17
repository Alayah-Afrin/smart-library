# How to test Smart Library V2

Two parts: **automatic tests** (the computer checks hundreds of things for you) and a
**hand checklist** (you click through the site and compare with what you should see).
If anything doesn't match, send a screenshot and the step number.

---

## Part 1 – Automatic tests (about 20 minutes in total)

Open a terminal in `D:\documents\Smart_Library_V2` and run these one at a time.
Each prints `PASS` / `FAIL` lines and a total at the end.

| Command | What it checks | Good result |
|---|---|---|
| `python tests/check_search.py` | 19 core searches (names, typos, titles, categories) | `19/19 checks passed` |
| `python tests/test_features.py` | Every feature: filters, reader, PDFs, shelf, reviews, sentiment, admin, spam guard, "harr", short forms… | all checks passed, no `FAIL` lines |
| `python tests/evaluate.py` | Search quality on 44 test questions (score 0 to 1) | **Hybrid** line about **0.94** |
| `python tests/filter_matrix.py` | 560 combinations: 40 searches × every category × Free on/off | `crashes: 0`, `category ignored: 0`, `free filter ignored: 0`, `confusing` ≤ 1 |

Any line starting with `FAIL` names exactly what broke.

**Maintenance (run now and then):**
- `python scripts/check_pdfs.py` – re-checks which textbook PDFs still open (publishers move files).

---

## Part 2 – Check it by hand in the browser

Start the site:

```
python app.py
```

Wait for `Running on http://127.0.0.1:5000` (about a minute), then open that address.
If a page ever looks like an *old* version, an old copy of the app is still running in another
window – close it and start again.

### Searching

| # | Do this | You should see |
|---|---|---|
| 1 | Search **harr** | Books by the 3 real authors named Harr (e.g. *A Civil Action*, Jonathan Harr) **and** a line "Did you mean **harry**?" |
| 2 | Click **harry** in that line | Harry Potter books (the link searches on our own site) |
| 2b | Search **hobb** | Robin Hobb's books and "Did you mean **hobbit**?" |
| 3 | Search **harry pot** | Harry Potter books (an unfinished word is completed) |
| 4 | Search **tolk**, then **gats**, then **calc** | Tolkien; The Great Gatsby; books starting with "calc…" |
| 5 | Search **jk**, then **jrr tolkien** | J.K. Rowling's books; Tolkien's books |
| 6 | Search **king**, **twain**, **garcia marquez** | Stephen King first; Mark Twain; Gabriel García Márquez |
| 7 | Search **shakspeare**, **moby dik**, **harry poter** | "Did you mean Shakespeare?"; Moby-Dick; Harry Potter |
| 8 | Search **introduction to physics** | Physics books – **not** "Metaphysics" |
| 9 | Search **ai**, **ml**, **dsa**, **lotr** | "Showing results for artificial intelligence (AI)", machine learning, data structures, The Lord of the Rings |
| 10 | Search **learn python** | 2 exact titles, then a "More books like this" section filling the page |
| 11 | Search **project hail mary** | "isn't in our library yet. These are the closest matches." |
| 12 | Search **asdfghjk** | "No books match…" and **no** random books |

### Filters

| # | Do this | You should see |
|---|---|---|
| 13 | **harry potter** + Category **Mathematics** | "No Mathematics books match…" and a link to search all categories |
| 14 | **harry** + Category **Science** | Real science books by authors named Harry (no science fiction) |
| 15 | **world war 2** + Category **Fiction** | No "Nonfiction" books |
| 16 | **charles dickens** + tick **Free to read** | Every card has a green **FREE** badge and **Read here** |

### Cards and pages

| # | Do this | You should see |
|---|---|---|
| 17 | Look at any row of result cards | All the same height, nothing sticking out, **one** main button each, a **Summary** button and a ♡ |
| 18 | Home page **Today's read** | Title, author, a short summary **if we have one** (otherwise a small "About: …" line), Read button and ♡ – **no** side panel |
| 19 | Click **Summary** on a card | A small box with the summary and the reviews |

### Reading inside the site

| # | Do this | You should see |
|---|---|---|
| 20 | Search **charles dickens** → **Read here** | The book opens in pages (A− / A+ / dark mode, ← → keys). If Gutenberg is down you get a polite message with a link instead – that's expected |
| 21 | Search **pro git** (or **college algebra**) → **Read here** | The textbook PDF opens **inside our page**. The first time can take up to a minute for big books; after that it's instant |
| 22 | Search **linear algebra** with **Free to read** | Textbooks: some say **Read here** (PDF opens on our site), others **Read free ↗** (opens the publisher's page) |

### Shelf, reviews, sentiment

| # | Do this | You should see |
|---|---|---|
| 23 | Search **harry potter**, click ♡ on a book, click **My shelf**, then the **Search** tab | Back on your Harry Potter results, ♥ still filled |
| 24 | Click the **Smart Library** title | A fresh home page |
| 25 | **Summary** on any book → 4★ + comment *"Loved it!"* → Post | "4.0★ from 1 reader", "Mostly positive", a 😊 tag on your comment |
| 26 | Post 5★ with *"Terrible, waste of time"* on another book | Saved; it will show in the admin page as **stars ≠ words** |
| 26b | Under a comment, click **Report** | Four choices appear: Spam · Offensive · Spoiler · Other (and Cancel). Pick one → "Thanks, reported as …". Reporting again → "You already reported this." |
| 27 | (Optional) start with an admin password: `$env:ADMIN_PASSWORD = "pick-one"` then `python app.py`, open **/admin** | Sign in, see "How readers feel", the flagged review, and "Reported as: 1× spam"; **Keep** / **Delete** work |

### Night mode

| # | Do this | You should see |
|---|---|---|
| 27b | Click the 🌙 button in the top menu | The whole site turns dark (cards, search box, summary box, reviews); the button becomes ☀️ |
| 27c | Reload the page, open My shelf, open a book with **Read here** | Still dark everywhere – the choice is remembered |
| 27d | Click ☀️ | Back to day mode |

### Phone size

| # | Do this | You should see |
|---|---|---|
| 28 | Make the browser window narrow (like a phone) | Cards stack, no sideways scrolling, Summary opens as a side panel |

Steps 25–26 save real reviews in `instance/community.sqlite3` – that's fine for testing.
