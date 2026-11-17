# Recipe Hugging Face uses to build and run the Smart Library website.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf-cache PORT=7860

# 1. libraries (CPU-only torch: no GPU on the free tier, and ~4x smaller)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r /tmp/requirements.txt

# 2. download the two AI models now, so visitors never wait for them
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')" \
 && chmod -R a+rwX /opt/hf-cache

# 3. the app itself, run as a normal user (Hugging Face expects user id 1000)
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . /app
USER user

EXPOSE 7860
# one worker (the search system needs ~3 GB of memory), several threads for visitors at the same time;
# long timeout because the first start downloads the book data
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "900", "--bind", "0.0.0.0:7860", "app:app"]
