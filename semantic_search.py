#!/usr/bin/env python3
"""
Semantic search over transcript chunks: embed a query with Fireworks (Qwen3),
then find the closest chunks in the DB and print episode + highlighted text.

Example of using Fireworks Embeddings API with qwen3-embedding-8b (768-d).
Requires: FIREWORKS_API_KEY, DB_USER, DB_PASSWORD, DB_HOST, DB_NAME.
"""
import argparse
import os
import sys

import requests
from sqlalchemy import create_engine, text


# Fireworks Embeddings API (OpenAI-compatible)
FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "fireworks/qwen3-embedding-8b"
QUERY_PREFIX = "search_query: "  # recommended for retrieval with this model


def get_embedding(query: str, api_key: str, model: str = DEFAULT_EMBEDDING_MODEL, dimensions: int = 768) -> list[float]:
    """Call Fireworks Embeddings API; return the 768-d vector for the query."""
    payload = {
        "model": model,
        "input": QUERY_PREFIX + query[:500],
        "dimensions": dimensions,
    }
    resp = requests.post(
        FIREWORKS_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


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


def search(engine, embedding: list[float], podcast_id: int | None, limit: int, schema: str):
    """Return rows: episode id/name/date, podcast_name, highlighted_content, rank."""
    vec_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    # pgvector expects a quoted string literal: '[...]'::vector
    vec_literal = "'" + vec_str.replace("'", "''") + "'::vector"

    sql = text(
        f"""
        SELECT
            e.id AS episode_id,
            e.name AS episode_name,
            e.date_published,
            p.name AS podcast_name,
            tc.timemark || ' -- ' || tc.content AS highlighted_content,
            1.0 / (1.0 + (tc.embedding <-> {vec_literal})) AS rank
        FROM transcript_chunks tc
        JOIN episodes e ON e.id = tc.episode_id
        LEFT JOIN podcasts p ON p.id = e.podcast_id
        WHERE tc.embedding IS NOT NULL
        AND (:podcast_id IS NULL OR e.podcast_id = :podcast_id)
        ORDER BY tc.embedding <-> {vec_literal}
        LIMIT :lim
        """
    )
    with engine.connect() as conn:
        # Include public so pgvector's "vector" type is visible (extension usually in public)
        conn.execute(text(f"SET search_path TO {schema}, public"))
        result = conn.execute(sql, {"podcast_id": podcast_id, "lim": limit})
        return result.mappings().fetchall()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search: embed query with Fireworks, return top matching transcript chunks."
    )
    parser.add_argument("query", help="Search query (embedded and matched against chunk embeddings)")
    parser.add_argument("--podcast-id", type=int, default=None, help="Restrict to this podcast ID (default: all)")
    parser.add_argument("-n", "--limit", type=int, default=20, help="Max results (default 20)")
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Set FIREWORKS_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    print("Embedding query with Fireworks (qwen3-embedding-8b)...", file=sys.stderr)
    embedding = get_embedding(args.query, api_key)
    engine = get_engine()
    schema = os.getenv("DB_SCHEMA", "whisper")
    rows = search(engine, embedding, args.podcast_id, args.limit, schema)

    if not rows:
        print("No matching chunks found.", file=sys.stderr)
        return

    print(f"\nTop {len(rows)} matches (query: {args.query!r})\n", file=sys.stderr)
    for i, row in enumerate(rows, 1):
        ep_name = row["episode_name"] or "(no name)"
        date_str = str(row["date_published"]) if row["date_published"] else "—"
        podcast = row["podcast_name"] or "—"
        content = (row["highlighted_content"] or "").strip()
        rank = row["rank"]
        print(f"{i}. [{podcast}] {ep_name} ({date_str}) [rank={rank:.4f}]")
        print(f"   {content}")
        print()


if __name__ == "__main__":
    main()
