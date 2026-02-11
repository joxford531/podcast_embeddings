# podcast_embeddings

Minimal repo for the topic extraction script: clusters transcript chunks by embedding, labels clusters via Fireworks LLM, prints top topics to stdout.

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

For all options, env vars, and behavior, see **`docs/`**:

- **`docs/TOPIC_EXTRACTION_SCRIPT.md`** — How to run, all flags, examples.
- **`docs/TOPIC_EXTRACTION.md`** — How it works (K-means, two-stage labeling).
