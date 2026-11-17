"""Make sure the book data is on disk before the search system loads.

On your laptop the files are already in data/processed, so nothing happens.
On the website (Hugging Face Space) the folder starts empty: the files are downloaded once from a
private Hugging Face *dataset* repo named in the DATA_REPO setting (for example "your-name/smart-library-data"),
using the HF_TOKEN secret. Upload them with:  python scripts/upload_data.py your-name/smart-library-data
"""
import os
import sys

DATA_DIR = os.path.join('data', 'processed')

# the files the website actually reads (see search.load_search_system)
REQUIRED = [
    'books_with_mood.parquet', 'books_search_ready.parquet', 'book_embedding.npy',
    'ratings_clean.parquet', 'gutenberg_links.parquet', 'extra_books.parquet', 'extra_embeddings.npy',
]


def missing_files(data_dir=DATA_DIR):
    return [f for f in REQUIRED if not os.path.exists(os.path.join(data_dir, f))]


def ensure_data(data_dir=DATA_DIR):
    missing = missing_files(data_dir)
    if not missing:
        return
    repo = os.environ.get('DATA_REPO', '').strip()
    if not repo:
        sys.exit(f"Book data missing ({', '.join(missing)}). Put the files in {data_dir}, "
                 f"or set DATA_REPO to the Hugging Face dataset that holds them.")
    from huggingface_hub import snapshot_download
    print(f"Downloading book data from the dataset {repo} (first start only, ~700 MB)...", flush=True)
    snapshot_download(repo_id=repo, repo_type='dataset', local_dir='data',
                      allow_patterns=[f'processed/{f}' for f in REQUIRED],
                      token=os.environ.get('HF_TOKEN') or None)
    still = missing_files(data_dir)
    if still:
        sys.exit(f"Download finished but these files are still missing: {', '.join(still)}")
    print("Book data ready.", flush=True)
