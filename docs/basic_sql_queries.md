# Basic SQL Queries for Podcast Transcripts

Simple examples using the `whisper` schema. Tables you care about here:

- **podcasts** — one row per podcast (id, name, …).
- **episodes** — one row per episode (id, podcast_id, name, date_published, …).
- **transcript_lines** — one row per line of transcript (episode_id, timemark, transcription).
- **transcriptions** — one row per episode, full transcript text plus a search index (episode_id, transcription, ts).

---

## 1. How many times does "chili" appear per episode? (one podcast, exact word)

You want: for a given podcast, each episode and how many times the word "chili" shows up in that episode’s transcript. We use **transcript_lines**: each row is one line; we count lines that contain "chili" (and optionally total occurrences in the line).

**Count of lines that contain "chili" per episode** (simplest):

```sql
SET search_path TO whisper;

SELECT
  e.id AS episode_id,
  e.name AS episode_name,
  e.date_published,
  COUNT(*) AS lines_containing_chili
FROM episodes e
JOIN transcript_lines tl ON tl.episode_id = e.id
WHERE e.podcast_id = 1   -- replace 1 with your podcast id
  AND LOWER(tl.transcription) LIKE '%chili%'
GROUP BY e.id, e.name, e.date_published
ORDER BY lines_containing_chili DESC, e.date_published DESC;
```

**Total number of times "chili" appears per episode** (if a line says "chili" twice, that counts as 2):

```sql
SET search_path TO whisper;

SELECT
  e.id AS episode_id,
  e.name AS episode_name,
  e.date_published,
  SUM(
    (LENGTH(tl.transcription) - LENGTH(REPLACE(LOWER(tl.transcription), 'chili', '')))
    / 5
  )::int AS chili_occurrence_count
FROM episodes e
JOIN transcript_lines tl ON tl.episode_id = e.id
WHERE e.podcast_id = 1   -- replace 1 with your podcast id
GROUP BY e.id, e.name, e.date_published
HAVING SUM(
  (LENGTH(tl.transcription) - LENGTH(REPLACE(LOWER(tl.transcription), 'chili', '')))
  / 5
) > 0
ORDER BY chili_occurrence_count DESC, e.date_published DESC;
```

(`/ 5` is because the word `"chili"` is 5 characters; each occurrence adds 5 to the length difference.)

---

## 2. Natural-language search: phrase like "Joey chili peanuts" (whole episodes)

Here you want to find **episodes** whose full transcript matches a phrase in a more “search engine” way: word order and phrasing matter, but the database uses full-text search so small variations (e.g. “chili” vs “chilies”) can still match. The **transcriptions** table has one row per episode and a pre-built search column `ts` (tsvector), so the query uses that instead of scanning line-by-line. This is the same pattern used for natural-language search in the web app.

**Find episodes containing the phrase (full-text search on whole transcript):**

```sql
SET search_path TO whisper;

SELECT
  e.id AS episode_id,
  e.name AS episode_name,
  e.date_published,
  p.name AS podcast_name,
  ts_headline(
    'english',
    t.transcription,
    websearch_to_tsquery('english', 'Joey chili peanuts'),
    'StartSel=<mark>,StopSel=</mark>,MaxWords=150,MaxFragments=3,FragmentDelimiter=" ... "'
  ) AS highlighted_content,
  ts_rank_cd(t.ts, websearch_to_tsquery('english', 'Joey chili peanuts')) AS rank
FROM episodes e
JOIN transcriptions t ON t.episode_id = e.id
LEFT JOIN podcasts p ON p.id = e.podcast_id
WHERE t.ts @@ websearch_to_tsquery('english', 'Joey chili peanuts')
  AND e.podcast_id = 1   -- replace 1 with your podcast id, or drop this line to search all podcasts
ORDER BY rank DESC, e.date_published DESC
LIMIT 25;
```

- `websearch_to_tsquery('english', 'Joey chili peanuts')` turns your phrase into a full-text query.
- `t.ts` is the pre-built tsvector on the full transcript; `@@` means "this episode matches the query."
- `ts_headline(...)` returns snippets of the transcript with matches wrapped in `<mark>...</mark>` (up to 3 fragments, 150 words total).
- `ts_rank_cd(t.ts, ...)` scores how well the episode matches so you can order by relevance.


---

## Quick reference

| Goal | Table | Idea |
|------|--------|------|
| Count lines containing "chili" per episode (one podcast) | transcript_lines | `LOWER(transcription) LIKE '%chili%'` + `GROUP BY episode` |
| Count total occurrences of "chili" per episode | transcript_lines | `LENGTH - REPLACE` trick, then `SUM` and `GROUP BY episode` |
| Search phrase "Joey chili peanuts" (natural language) | transcriptions | `t.ts @@ websearch_to_tsquery('english', '...')`, `ts_headline` for snippets, `ts_rank_cd` for rank |
