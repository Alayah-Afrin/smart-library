"""Upload the book data the website needs to a PRIVATE Hugging Face dataset repo.

Run from D:/documents/Smart_Library_V2 (only after you've decided to publish):
    python scripts/upload_data.py your-name/smart-library-data
It uses your Hugging Face login: run  hf auth login  first (you paste your own token there).
Only the 7 files listed in src/fetch_data.py are uploaded (about 680 MB); the repo is created private.
"""
import os
import sys

sys.path.insert(0, 'src')
from fetch_data import DATA_DIR, REQUIRED, missing_files

if len(sys.argv) != 2 or '/' not in sys.argv[1]:
    sys.exit('Usage: python scripts/upload_data.py your-name/smart-library-data')
repo = sys.argv[1]

missing = missing_files()
if missing:
    sys.exit(f'These files are missing locally, nothing uploaded: {missing}')

from huggingface_hub import HfApi

api = HfApi()
total = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in REQUIRED) / 1e6
print(f'Uploading {len(REQUIRED)} files ({total:.0f} MB) to the PRIVATE dataset {repo} ...')
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=DATA_DIR, path_in_repo='processed',
                  allow_patterns=REQUIRED, commit_message='Book data for Smart Library')
print('Done. In the Space settings set  DATA_REPO =', repo, ' and add your token as the secret HF_TOKEN.')
