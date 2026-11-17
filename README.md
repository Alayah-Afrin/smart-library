---
title: Smart Library
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid AI book search with 65,000+ free legal books
---

# Smart Library

A book search and reading website for students. It understands what you mean, not just the exact words you type,
and lets you read 65,000+ free, legal books right on the site.

## What it does

**Search that understands you**
- **Hybrid search:** keyword search (BM25) + meaning search (sentence embeddings, FAISS), combined.
  On 44 test questions: BM25 alone nDCG@10 = 0.75, meaning search alone = 0.68, **hybrid = 0.94**
  (`python tests/evaluate.py`).
- Typos (*roling, harry poter, moby dik*), unfinished words (*harry pot, tolk*), initials (*jk, jrr*),
  study short forms (*ai, ml, dsa, dbms*), "Did you mean …?" links.
- Honest messages when nothing matches, filters (category, free to read) that always apply.

**Free books, read on the site**
- 65,000+ public-domain and openly licensed books (Project Gutenberg, Open Textbook Library, OpenStax).
- In-site reader with pages, text size and night mode; textbook PDFs open inside the page.

**Readers**
- My shelf, reading streak and "Continue reading" (no login needed).
- Ratings and comments with AI sentiment analysis (RoBERTa), report with a reason, admin moderation.

## Run it on your computer

```
pip install torch
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. The book data must be in `data/processed` (see `src/fetch_data.py`).

## Settings (environment variables)

| Name | What it does |
|---|---|
| `SECRET_KEY` | Signs browser sessions. Set a long random value on the website. |
| `ADMIN_PASSWORD` | Turns on the `/admin` moderation page. |
| `DATA_REPO` | Hugging Face dataset holding the book data, downloaded on first start. |
| `HF_TOKEN` | Token that can read that (private) dataset. |
| `PORT` | Port for `python app.py` (default 5000). |

## Tests

See [TESTING.md](TESTING.md): automatic tests plus a click-through checklist.
