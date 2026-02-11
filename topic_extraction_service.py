#!/usr/bin/env python3
"""
Topic extraction from transcript_chunks embeddings (read-only).
Clusters with K-means, labels clusters via Fireworks, prints top N topics to stdout.

Requires: FIREWORKS_API_KEY, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_SCHEMA (optional)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import requests
from sqlalchemy import create_engine, nulls_last, text
from sqlalchemy.orm import sessionmaker
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from orm_classes import Episode, TranscriptChunk


def get_engine():
    db_user = os.getenv("DB_USER", "app")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", "5432"))
    app_db = os.getenv("DB_NAME", "app")
    if not all([db_user, db_password, db_host, app_db]):
        raise SystemExit(
            "Missing DB env: set DB_USER, DB_PASSWORD, DB_HOST, DB_NAME (and optionally DB_PORT, DB_SCHEMA)"
        )
    return create_engine(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{app_db}"
    )


def load_embeddings(engine, podcast_id=None, first_n_episodes=None, last_n_episodes=None):
    app_schema = os.getenv("DB_SCHEMA", "whisper")
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.execute(text(f"SET search_path TO {app_schema};"))
        if first_n_episodes is not None:
            # First N episodes by id (optionally for this podcast)
            ep_q = session.query(Episode.id).order_by(Episode.id.asc())
            if podcast_id is not None:
                ep_q = ep_q.filter(Episode.podcast_id == podcast_id)
            episode_ids = [r[0] for r in ep_q.limit(first_n_episodes).all()]
            if not episode_ids:
                return [], np.array([])
            q = (
                session.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.embedding.isnot(None),
                    TranscriptChunk.episode_id.in_(episode_ids),
                )
            )
        elif last_n_episodes is not None:
            # Most recent N episodes (by date_published desc, then id desc)
            ep_q = (
                session.query(Episode.id)
                .order_by(nulls_last(Episode.date_published.desc()), Episode.id.desc())
            )
            if podcast_id is not None:
                ep_q = ep_q.filter(Episode.podcast_id == podcast_id)
            episode_ids = [r[0] for r in ep_q.limit(last_n_episodes).all()]
            if not episode_ids:
                return [], np.array([])
            q = (
                session.query(TranscriptChunk)
                .filter(
                    TranscriptChunk.embedding.isnot(None),
                    TranscriptChunk.episode_id.in_(episode_ids),
                )
            )
        else:
            q = session.query(TranscriptChunk).filter(TranscriptChunk.embedding.isnot(None))
            if podcast_id is not None:
                q = q.join(Episode).filter(Episode.podcast_id == podcast_id)
        chunks = q.all()
    if not chunks:
        return [], np.array([])
    embeddings = np.array([c.embedding for c in chunks])
    return chunks, embeddings


def prepare_embeddings(embeddings, normalize_l2=False, scale=True):
    """Optionally L2-normalize and/or standard-scale embeddings before clustering."""
    X = np.asarray(embeddings, dtype=np.float64)
    if normalize_l2:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        X = X / norms
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X


def choose_k(
    embeddings,
    min_clusters,
    max_clusters,
    normalize_l2=False,
    scale=True,
    n_init=10,
    max_iter=300,
    step=None,
):
    X = prepare_embeddings(embeddings, normalize_l2=normalize_l2, scale=scale)
    n = len(X)
    max_clusters = min(max_clusters, max(2, n // 5))
    min_clusters = max(min_clusters, min(max_clusters, n // 400))
    min_clusters = min(min_clusters, max_clusters)
    if step is None:
        step = max(1, (max_clusters - min_clusters) // 20)
    best_k, best_score = min_clusters, -1.0
    for k in range(min_clusters, max_clusters + 1, step):
        try:
            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=n_init,
                max_iter=max_iter,
            )
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k


def cluster(embeddings, n_clusters, normalize_l2=False, scale=True, n_init=10, max_iter=300):
    X = prepare_embeddings(embeddings, normalize_l2=normalize_l2, scale=scale)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = km.fit_predict(X)
    return labels, km, X


PODCAST_FILLER_STOP = frozenset({
    "yeah", "just", "don", "don't", "know", "oh", "got", "ve", "ha", "really", "right",
    "okay", "like", "thing", "things", "fucking", "shit", "gonna", "wanna", "kinda",
    "sorta", "actually", "literally", "basically", "maybe", "probably", "think", "thought",
    "say", "said", "says", "mean", "means", "go", "going", "come", "get", "gets",
    "make", "makes", "take", "see", "look", "looks", "way", "lot", "bit",
    "something", "anything", "everything", "nothing", "someone", "everyone", "cause",
    "because", "though", "pretty", "much", "well", "so", "um", "uh", "hmm", "hey",
    "yo", "nah", "nope", "yep", "yup", "ok", "yes", "no",
})
EXTENDED_STOP_WORDS = set(ENGLISH_STOP_WORDS) | PODCAST_FILLER_STOP


def extract_cluster_keywords(chunks, labels, n_keywords=10):
    vectorizer = TfidfVectorizer(
        stop_words=list(EXTENDED_STOP_WORDS),
        max_features=5000,
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform([c.content for c in chunks])
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for cid in set(int(l) for l in labels):
        mask = [i for i, l in enumerate(labels) if int(l) == cid]
        avg_tfidf = tfidf_matrix[mask].mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[::-1]
        chosen = []
        for idx in top_indices:
            if len(chosen) >= n_keywords:
                break
            w = terms[idx]
            if len(w) >= 3 and w not in EXTENDED_STOP_WORDS:
                chosen.append(w)
        cluster_keywords[cid] = chosen[:n_keywords] if chosen else [terms[top_indices[0]]]
    return cluster_keywords


def _extract_topic_from_response(content: str) -> str:
    if not content:
        return ""
    think_pat = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    text = think_pat.sub("", content)
    for line in text.splitlines():
        line = line.strip().strip("'\"").strip()
        if not line:
            continue
        if "topic name" in line.lower() or "user provided" in line.lower():
            continue
        if line.lower().startswith("<think>") or line.lower().startswith("</think>"):
            continue
        if line.lower() in ("<think>", "</think>"):
            continue
        label = re.sub(r"^<think>\s*", "", line, flags=re.IGNORECASE)
        label = re.sub(r"\s*</think>\s*$", "", label, flags=re.IGNORECASE)
        if label and label.lower() not in ("<think>", "</think>"):
            return label[:80]
    return ""


def _episode_diverse_sample_indices(chunks, indices, X, centroid, n=6, max_per_episode=2, prefer_light_episodes=False):
    if not indices or n <= 0:
        return []
    by_episode = {}
    for i in indices:
        ep_id = chunks[i].episode_id
        by_episode.setdefault(ep_id, []).append(i)
    idx_arr = np.asarray(indices)
    pts = X[idx_arr]
    dists = np.linalg.norm(pts - centroid, axis=1)
    idx_to_dist = dict(zip(indices, dists))
    for ep_id in by_episode:
        by_episode[ep_id].sort(key=lambda idx: idx_to_dist[idx])
    if prefer_light_episodes:
        episode_order = sorted(by_episode.keys(), key=lambda ep_id: len(by_episode[ep_id]))
    else:
        episode_order = sorted(
            by_episode.keys(),
            key=lambda ep_id: idx_to_dist[by_episode[ep_id][0]],
        )
    chosen = []
    for ep_id in episode_order:
        if len(chosen) >= n:
            break
        take = min(max_per_episode, n - len(chosen), len(by_episode[ep_id]))
        chosen.extend(by_episode[ep_id][:take])
    return chosen[:n]


def label_cluster(fireworks_api_key, cluster_chunks, cluster_id, base_url, model, keywords=None, max_chars=2000):
    if not cluster_chunks:
        return f"Topic {cluster_id}", False
    sample = list(cluster_chunks)
    text_blob = "\n\n".join(c.content for c in sample)
    if len(text_blob) > max_chars:
        text_blob = text_blob[:max_chars] + "..."
    keywords_str = ", ".join(keywords) if keywords else "(none)"
    prompt = f"""This podcast topic cluster has these distinguishing keywords (ranked by statistical importance across all chunks in the cluster):
{keywords_str}

Here are a few representative excerpts for context:
{text_blob}

Based on the keywords and excerpts, give a short topic name (2–6 words) that captures what this cluster is about. The name MUST reflect the keywords — do not name the topic after a detail, person, or phrase that does not appear in the keyword list. Use plain, recognizable terms.

Reply with only the topic name, nothing else."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fireworks_api_key}",
    }
    for payload in (
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 40, "reasoning_effort": "none"},
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 40},
    ):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            label = _extract_topic_from_response(content).strip().strip("'\"").strip()
            if not label:
                return f"Topic {cluster_id}", False
            return label, True
        except requests.RequestException:
            if "reasoning_effort" in payload:
                continue
            print(f"Warning: could not label cluster {cluster_id}", file=sys.stderr)
            return f"Topic {cluster_id}", False
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: could not parse response for cluster {cluster_id}: {e}", file=sys.stderr)
            return f"Topic {cluster_id}", False
    print(f"Warning: could not label cluster {cluster_id} (API rejected request)", file=sys.stderr)
    return f"Topic {cluster_id}", False


def _parse_args():
    parser = argparse.ArgumentParser(description="Extract top N topics from transcript_chunks (read-only, prints to stdout)")
    parser.add_argument("-n", "--top", type=int, default=20, help="Number of top topics to print (default 20)")
    parser.add_argument("--podcast-id", type=int, default=None, help="Restrict to this podcast ID")
    parser.add_argument("--n-clusters", type=int, default=None, help="Fixed number of clusters (default: auto)")
    parser.add_argument("--min-clusters", type=int, default=5, help="Min clusters when auto (default 5)")
    parser.add_argument("--max-clusters", type=int, default=50, help="Max clusters when auto (default 50)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before clustering")
    parser.add_argument("--no-scale", action="store_true", help="Skip StandardScaler")
    parser.add_argument("--first-n-episodes", type=int, default=None, metavar="N", help="Only chunks from first N episodes (by episode id)")
    parser.add_argument("--last-n-episodes", type=int, default=None, metavar="N", help="Only chunks from most recent N episodes (by date_published desc)")
    parser.add_argument("--fireworks-base-url", type=str, default=os.getenv("FIREWORKS_API_BASE_URL", "https://api.fireworks.ai/inference/v1"))
    parser.add_argument("--fireworks-model", type=str, default=os.getenv("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/qwen3-235b-a22b"))
    parser.add_argument("--semantic", action="store_true", help="L2-normalize + no scaling (semantic clustering)")
    parser.add_argument("--n-init", type=int, default=10, metavar="N")
    parser.add_argument("--max-iter", type=int, default=300, metavar="N")
    parser.add_argument("--k-step", type=int, default=None, metavar="N")
    parser.add_argument("--min-chunks-per-episode", type=int, default=1, metavar="N")
    parser.add_argument("--rank-by", choices=("chunks", "episodes"), default="chunks")
    parser.add_argument("--show-keywords", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if not os.getenv("FIREWORKS_API_KEY"):
        raise SystemExit("Set FIREWORKS_API_KEY in the environment")
    if args.first_n_episodes is not None and args.last_n_episodes is not None:
        raise SystemExit("Use only one of --first-n-episodes or --last-n-episodes")


def _run_clustering(chunks, embeddings, args):
    if args.semantic:
        args.normalize = True
        args.no_scale = True
    scale = not args.no_scale
    if args.normalize:
        print("Using L2-normalized embeddings", file=sys.stderr)
    if args.no_scale:
        print("Skipping StandardScaler", file=sys.stderr)
    n_clusters = args.n_clusters
    if n_clusters is None:
        n_clusters = choose_k(embeddings, args.min_clusters, args.max_clusters, normalize_l2=args.normalize, scale=scale, n_init=args.n_init, max_iter=args.max_iter, step=args.k_step)
        print(f"Auto chose n_clusters={n_clusters}", file=sys.stderr)
    labels, km, X = cluster(embeddings, n_clusters, normalize_l2=args.normalize, scale=scale, n_init=args.n_init, max_iter=args.max_iter)
    return labels, km, X


def _build_cluster_index(chunks, labels, min_chunks_per_episode):
    by_cluster = {}
    cluster_indices = {}
    for i, c in enumerate(chunks):
        cid = int(labels[i])
        by_cluster.setdefault(cid, []).append(c)
        cluster_indices.setdefault(cid, []).append(i)
    def distinct_episode_count(cid):
        counts = Counter(c.episode_id for c in by_cluster[cid])
        return sum(1 for n in counts.values() if n >= min_chunks_per_episode)
    return by_cluster, cluster_indices, distinct_episode_count


def _select_top_clusters(by_cluster, distinct_episode_count, rank_by, top_n):
    if rank_by == "chunks":
        return sorted(by_cluster.keys(), key=lambda c: len(by_cluster[c]), reverse=True)[:top_n]
    return sorted(by_cluster.keys(), key=distinct_episode_count, reverse=True)[:top_n]


def _label_top_clusters(api_key, chunks, sorted_clusters, by_cluster, cluster_indices, cluster_keywords, distinct_episode_count, km, X, args):
    print("Labeling top topics via Fireworks (TF-IDF keywords + excerpts)...", file=sys.stderr)
    topic_names = {}
    for cid in sorted_clusters:
        centroid = km.cluster_centers_[cid]
        episode_count = distinct_episode_count(cid)
        if episode_count >= 50:
            n_sample, max_chars, max_per_episode = 6, 2000, 1
        elif episode_count >= 25:
            n_sample, max_chars, max_per_episode = 5, 1800, 1
        else:
            n_sample, max_chars, max_per_episode = 4, 1500, 2
        prefer_light = episode_count >= 25
        sample_idx = _episode_diverse_sample_indices(chunks, cluster_indices[cid], X, centroid, n=n_sample, max_per_episode=max_per_episode, prefer_light_episodes=prefer_light)
        sample_chunks = [chunks[i] for i in sample_idx]
        name, _ = label_cluster(api_key, sample_chunks, cid, base_url=args.fireworks_base_url, model=args.fireworks_model, keywords=cluster_keywords.get(cid, []), max_chars=max_chars)
        topic_names[cid] = name
    return topic_names


def _print_results(sorted_clusters, by_cluster, distinct_episode_count, topic_names, cluster_keywords, chunks, args):
    total_chunks = len(chunks)
    total_episodes = len({c.episode_id for c in chunks})
    min_chunks_per_episode = args.min_chunks_per_episode
    rank_label = "chunk count" if args.rank_by == "chunks" else "episode count"
    ep_note = f" (episode = ≥{min_chunks_per_episode} chunks in cluster)" if min_chunks_per_episode > 1 else ""
    print(f"Total chunks: {total_chunks} across {total_episodes} episodes\nTop {args.top} topics (ranked by {rank_label}){ep_note}\n")
    for rank, cid in enumerate(sorted_clusters, 1):
        chunk_count = len(by_cluster[cid])
        episode_count = distinct_episode_count(cid)
        pct_chunks = 100.0 * chunk_count / total_chunks
        name = topic_names[cid]
        ep_str = f"{episode_count} eps (≥{min_chunks_per_episode} chunks)" if min_chunks_per_episode > 1 else f"{episode_count} eps"
        line = f"{rank}. {name} — {chunk_count} chunks ({pct_chunks:.1f}%) | {ep_str}"
        if args.show_keywords:
            kw = ", ".join(cluster_keywords.get(cid, []))
            line += f"\n   keywords: {kw}"
        print(line)


def main():
    args = _parse_args()
    _validate_args(args)
    engine = get_engine()
    chunks, embeddings = load_embeddings(engine, podcast_id=args.podcast_id, first_n_episodes=args.first_n_episodes, last_n_episodes=args.last_n_episodes)
    if len(chunks) < 2:
        print("Not enough chunks with embeddings", file=sys.stderr)
        sys.exit(1)
    labels, km, X = _run_clustering(chunks, embeddings, args)
    by_cluster, cluster_indices, distinct_episode_count = _build_cluster_index(chunks, labels, args.min_chunks_per_episode)
    sorted_clusters = _select_top_clusters(by_cluster, distinct_episode_count, args.rank_by, args.top)
    print("Extracting TF-IDF keywords per cluster...", file=sys.stderr)
    cluster_keywords = extract_cluster_keywords(chunks, labels, n_keywords=10)
    api_key = os.getenv("FIREWORKS_API_KEY")
    topic_names = _label_top_clusters(api_key, chunks, sorted_clusters, by_cluster, cluster_indices, cluster_keywords, distinct_episode_count, km, X, args)
    _print_results(sorted_clusters, by_cluster, distinct_episode_count, topic_names, cluster_keywords, chunks, args)


if __name__ == "__main__":
    main()
