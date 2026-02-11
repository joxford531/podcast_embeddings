# podcast_embeddings

Scripts for working with podcast transcript embeddings: **semantic search** (Fireworks + pgvector) and **topic extraction** (K-means + LLM labeling).

## Install

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
export FIREWORKS_API_KEY=...
export DB_USER=... DB_PASSWORD=... DB_HOST=... DB_NAME=...
python topic_extraction_service.py [options]
```

Example: last 100 episodes of podcast 1, top 20 topics:

```bash
python topic_extraction_service.py --podcast-id 1 --last-n-episodes 100
```

### Semantic search

Embed a query with Fireworks (Qwen3-embedding-8b), find closest transcript chunks, print episode and text (limit 20):

```bash
python semantic_search.py "your search query" [--podcast-id N] [-n 20]
```

Same env vars; optional `--podcast-id` to restrict to one podcast.

For all options, env vars, and behavior, see **`docs/`**:

- **`docs/TOPIC_EXTRACTION_SCRIPT.md`** — How to run, all flags, examples.
- **`docs/TOPIC_EXTRACTION.md`** — How it works (K-means, two-stage labeling).
