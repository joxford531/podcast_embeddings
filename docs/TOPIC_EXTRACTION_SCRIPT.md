# Topic Extraction Script — How to Run and All Flags

This document describes **`topic_extraction_service.py`**: how to run it, what each flag does, and how the script behaves end-to-end. For the concepts behind K-means and two-stage labeling, see **`docs/TOPIC_EXTRACTION.md`**.

---

## How the Script Runs

**Location:** `topic_extraction_service.py` at the **repository root** (this repo contains only this script and supporting files).

**Run from:** The **repository root** directory. The script imports `Episode` and `TranscriptChunk` from `orm_classes` in the same folder.

**Execution flow:**

1. **Parse and validate** — Read CLI flags; require `FIREWORKS_API_KEY`; disallow using both `--first-n-episodes` and `--last-n-episodes` together.
2. **Load data** — Connect to the DB (using env vars), set schema to `whisper` (or `DB_SCHEMA`), and load transcript chunks that have non-null embeddings. Optionally filter by podcast and/or episode range (first N or last N episodes).
3. **Cluster** — Optionally normalize/scale embeddings; choose K (auto via silhouette or fixed); run K-means. Each chunk gets a cluster ID.
4. **Index** — Build a mapping from cluster ID to list of chunks and list of chunk indices; define "episode count" for a cluster as number of episodes that have at least `--min-chunks-per-episode` chunks in that cluster.
5. **Select top clusters** — Sort clusters by chunk count or by episode count (depending on `--rank-by`) and keep the top `--top` (e.g. 20).
6. **Keywords** — Run TF-IDF on all chunk text (extended stop words, top 10 terms per cluster, length ≥ 3).
7. **Label** — For each of the top clusters, sample a few episode-diverse excerpts and call the Fireworks LLM with the cluster's keywords + excerpts; parse the response into a short topic name (strip `<think>` blocks and tag-only lines).
8. **Print** — Write to stdout a header (total chunks, total episodes) and one line per topic: rank, name, chunk count, percentage, episode count; if `--show-keywords`, append the TF-IDF keywords for that topic.

**Database:** Read-only. The script only reads from `whisper.transcript_chunks` and (for episode filters) `whisper.episodes`. It does not create or write to any tables.

**Environment variables:**

| Variable | Required | Purpose |
|----------|----------|---------|
| `FIREWORKS_API_KEY` | Yes | Used for the labeling API calls. |
| `DB_USER` | Yes | PostgreSQL user. |
| `DB_PASSWORD` | Yes | PostgreSQL password. |
| `DB_HOST` | Yes | PostgreSQL host. |
| `DB_NAME` | Yes | PostgreSQL database name (e.g. `app`). |
| `DB_PORT` | No | Default `5432`. |
| `DB_SCHEMA` | No | Default `whisper`. Search path is set to this schema. |
| `FIREWORKS_API_BASE_URL` | No | Override for Fireworks API base URL (default `https://api.fireworks.ai/inference/v1`). |
| `FIREWORKS_CHAT_MODEL` | No | Override for the chat model ID (default `accounts/fireworks/models/qwen3-235b-a22b`). |

**Example (minimal):**

```bash
cd /path/to/podcast_embeddings   # repository root
export FIREWORKS_API_KEY=...
export DB_HOST=...
export DB_PASSWORD=...
python topic_extraction_service.py --podcast-id 1 --last-n-episodes 100
```

---

## All Flags (What Each One Does)

### Output and scope

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`-n`, `--top`** | integer | `20` | Number of topic lines to print. Only the top N clusters (by chunk count or episode count) are labeled and shown; the rest are skipped. |

### Data selection

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--podcast-id`** | integer | None | Restrict chunks to episodes that belong to this podcast ID. If omitted, all episodes (in the chosen episode range, if any) are used. |
| **`--first-n-episodes`** | integer | None | Use only chunks from the **first N episodes** when ordered by episode ID ascending. Useful for smaller/faster runs or for a fixed "slice" of the catalog. Cannot be used together with `--last-n-episodes`. |
| **`--last-n-episodes`** | integer | None | Use only chunks from the **most recent N episodes** when ordered by `date_published` descending (then `id` descending; null dates last). Useful to analyze only recent content. Cannot be used together with `--first-n-episodes`. |

### Clustering: number of clusters (K)

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--n-clusters`** | integer | None | **Fixed K.** If set, the script uses this many clusters and does not run the automatic K selection. If omitted, K is chosen automatically (see below). |
| **`--min-clusters`** | integer | `5` | When K is chosen automatically, the script tries K values between this and `--max-clusters`. Also used as a lower bound when a "floor" is applied for large datasets (e.g. at least `n_chunks // 400` clusters). |
| **`--max-clusters`** | integer | `50` | When K is chosen automatically, no K larger than this is tried. The effective maximum is also capped by data size (e.g. at most `n_chunks // 5`). |

When `--n-clusters` is not set, the script runs K-means for several K values (stepping from `--min-clusters` to `--max-clusters` with step size derived from the range) and picks the K with the best **silhouette score** (how well chunks sit in their own cluster vs. others).

### Clustering: embedding preprocessing

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--normalize`** | flag | off | L2-normalize each embedding vector before clustering. With normalized vectors, Euclidean distance behaves like cosine similarity, which is often better for semantic embeddings. |
| **`--no-scale`** | flag | off | Skip StandardScaler. By default the script scales each dimension to zero mean and unit variance after optional L2 normalization. Turning this off uses raw (or only L2-normalized) vectors. |
| **`--semantic`** | flag | off | **Preset:** turns on `--normalize` and turns on `--no-scale`. Recommended when you want clustering to reflect semantic (cosine-like) similarity. |

So you can get: default (scale only), L2 only (`--normalize --no-scale`), L2 + scale (`--normalize`), or raw (`--no-scale`). `--semantic` is equivalent to L2 only.

### Clustering: K-means tuning

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--n-init`** | integer | `10` | Number of times K-means runs with different initial centers; the best run (by inertia) is kept. Higher can improve stability at the cost of time. |
| **`--max-iter`** | integer | `300` | Maximum iterations per K-means run. The algorithm stops earlier if assignments stop changing. |
| **`--k-step`** | integer | None | When choosing K automatically, the script tries K = min, min+step, min+2*step, … up to max. If omitted, step is `(max - min) // 20`. Smaller step (e.g. `1`) gives a finer search but more runs. |

### Episode count and ranking

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--min-chunks-per-episode`** | integer | `1` | An episode is counted as "mentioning" a topic only if that cluster has **at least this many chunks** from that episode. Use `2` or `3` to avoid counting episodes that only barely touch a topic (e.g. one tangential chunk). The same threshold is used for ranking when `--rank-by episodes` and for the "episode count" shown in each line. |
| **`--rank-by`** | choice | `chunks` | How to order the top topics. `chunks` (default): sort by total chunk count in the cluster. `episodes`: sort by number of episodes that have ≥ `--min-chunks-per-episode` chunks in that cluster. In both cases the script prints chunk count and episode count per topic; only the **order** changes. |

### Labeling and debugging

| Flag | Type | Default | What it does |
|------|------|---------|--------------|
| **`--show-keywords`** | flag | off | For each printed topic, add a second line showing the TF-IDF keywords that were sent to the LLM for that cluster. Useful to see why a topic got its name or to debug weak keywords. |
| **`--fireworks-base-url`** | string | env or default | Override the base URL for the Fireworks API (e.g. chat completions). Default comes from `FIREWORKS_API_BASE_URL` or `https://api.fireworks.ai/inference/v1`. |
| **`--fireworks-model`** | string | env or default | Override the chat model used for labeling (e.g. `accounts/fireworks/models/qwen3-235b-a22b`). Default comes from `FIREWORKS_CHAT_MODEL` or that same model ID. |

---

## Example Invocations

**Last 150 episodes of podcast 1, only count episodes with ≥2 chunks per topic, semantic clustering:**

```bash
python topic_extraction_service.py --podcast-id 1 --last-n-episodes 150 --min-chunks-per-episode 2 --semantic
```

**First 200 episodes, fixed 25 clusters, show TF-IDF keywords:**

```bash
python topic_extraction_service.py --podcast-id 2 --first-n-episodes 200 --n-clusters 25 --show-keywords
```

**Rank by episode count instead of chunk count, top 15 topics:**

```bash
python topic_extraction_service.py --podcast-id 1 --last-n-episodes 100 --rank-by episodes -n 15
```

**Finer automatic K search (smaller step), more K-means restarts:**

```bash
python topic_extraction_service.py --podcast-id 1 --last-n-episodes 50 --k-step 1 --n-init 20
```

---

## Output Format

The script prints to **stdout**:

1. A line like: `Total chunks: 12805 across 300 episodes`
2. A line like: `Top 20 topics (ranked by chunk count) (episode = ≥2 chunks in cluster)` (the "episode = ≥N chunks" part appears only if `--min-chunks-per-episode` > 1).
3. One line per topic, e.g.  
   `1. Movie and TV discussion — 412 chunks (9.2%) | 45 eps`  
   If `--show-keywords` is set, each topic is followed by a line like:  
   `   keywords: movie, movies, film, watch, good, guy, ...`

Progress and warnings (e.g. "Auto chose n_clusters=36", "Labeling top topics…") go to **stderr**.

---

## Related Files

- **`topic_extraction_service.py`** — The script.
- **`orm_classes.py`** — Defines `TranscriptChunk`, `Episode` (and pgvector) in this repo.
- **`docs/TOPIC_EXTRACTION.md`** — Conceptual guide: K-means and two-stage labeling in plain English with diagrams.
