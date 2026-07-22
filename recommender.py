"""
Phase 2 — Movie Recommendation Engine
Netflix Recommender Project

Embedding strategy:
  Both movie chunks AND the user profile use the same 4-field template:
    Title: <title>
    Genres: <genres>
    Keywords: <keywords>
    Overview: <description>

  This ensures movie vectors and user profile vectors live in the same semantic
  space, making cosine similarity a meaningful apples-to-apples comparison.
  Director, Cast, and Tagline are intentionally excluded — they add noise
  without contributing to genre/tone/theme matching.
"""

from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
CHUNKS_CSV  = DATA_DIR / "final_movie_chunks_for_rag.csv"
EMBED_CACHE = DATA_DIR / "movie_embeddings.npy"
INDEX_CACHE = DATA_DIR / "movie_titles_index.json"

EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 5
COMFORT_POOL = 50   # broader pool for comfort-zone pass


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MovieHit:
    title: str
    genres: str
    overview: str
    similarity: float
    chunk_text: str       # full original chunk for RAG grounding (not shown to user)
    movie_id: int = 0
    vote_average: float = 0.0
    popularity: float = 0.0

@dataclass
class RecommendationSet:
    best_fit: list[MovieHit] = field(default_factory=list)
    comfort_zone: list[MovieHit] = field(default_factory=list)
    # Internal/debug — never shown to user
    user_query_best: str = ""
    user_query_comfort: str = ""
    embed_dim: int = 384


# ── Chunk field parser ────────────────────────────────────────────────────────

def _parse_field(chunk_text: str, field_name: str) -> str:
    """Extract a single labelled field value from chunk_text, e.g. 'Genres:'."""
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field_name}:"):
            return stripped[len(field_name) + 1:].strip()
    return ""


# ── Movie embed chunk builder ─────────────────────────────────────────────────

def build_movie_embed_chunk(chunk_text: str, title: str) -> str:
    """
    Build a focused 4-field embed chunk from raw chunk_text.

    Uses ONLY: Title, Genres, Keywords, Overview.
    Excludes Director, Cast, Tagline — these fields add retrieval noise
    without contributing meaningfully to theme/tone similarity.

    Returns:
        Title: <title>
        Genres: <genres>
        Keywords: <keywords>
        Overview: <overview>
    """
    genres   = _parse_field(chunk_text, "Genres")
    keywords = _parse_field(chunk_text, "Keywords")
    overview = _parse_field(chunk_text, "Overview")

    parts = [f"Title: {title}"]
    if genres:
        parts.append(f"Genres: {genres}")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if overview:
        parts.append(f"Overview: {overview}")

    return "\n".join(parts)


# ── User profile embed chunk builders ────────────────────────────────────────

def build_user_embed_chunk(result) -> str:
    """
    Build the best-fit user profile embed chunk using the same 4-field format
    as movie chunks (Title / Genres / Keywords / Overview), so that
    cosine similarity operates in a consistent semantic space.
    """
    prefs = result.movie_prefs
    mbti  = result.mbti

    # Personality → keyword signals that map to movie vocabulary
    sn_keywords = (
        "imagination, high-concept, surreal, symbolic, philosophical, abstract"
        if mbti.sn == "N" else
        "realistic, grounded, factual, practical, detail-oriented"
    )
    tf_keywords = (
        "cerebral, logical, clever, plot twist, intellectual"
        if mbti.tf == "T" else
        "emotional, heartfelt, empathy, relationships, values-driven"
    )
    jp_keywords = (
        "structured, plot-driven, resolution, clear narrative"
        if mbti.jp == "J" else
        "open-ended, spontaneous, ambiguous, experimental"
    )
    ei_keywords = (
        "ensemble cast, social dynamics, community, dialogue-heavy"
        if mbti.ei == "E" else
        "introspective, internal conflict, lone protagonist, quiet"
    )

    # Rec goal → keywords that match the movie Keywords vocabulary
    goal_keyword_map = {
        "Funny / Light-hearted":          "comedy, humor, funny, feel-good, lighthearted",
        "Emotional / Moving":             "emotional, heartfelt, tragedy, love, loss, moving",
        "Suspenseful / Tense":            "suspense, thriller, tension, mystery, danger",
        "Thought-provoking":              "thought-provoking, philosophical, mind-bending, reality, identity",
        "Relaxing / Comforting":          "comfort, warmth, feel-good, slice of life, relaxing",
        "Educational":                    "documentary, history, science, true story, informative",
        "Unique / Unconventional":        "experimental, arthouse, unconventional, surreal, original",
        "Featuring a specific actor or director": "acclaimed director, auteur, award-winning",
        "Other":                          "varied",
    }
    goal_kw = goal_keyword_map.get(prefs.next_rec_goal, prefs.next_rec_goal)

    all_keywords = ", ".join(filter(None, [
        goal_kw, sn_keywords, tf_keywords, jp_keywords, ei_keywords
    ]))

    # Genre field: user's stated genre + personality-adjacent tones
    genre_map = {
        "Action":       "Action, Thriller, Adventure",
        "Adventure":    "Adventure, Fantasy, Science Fiction",
        "Comedy":       "Comedy, Drama, Romance",
        "Drama":        "Drama, Romance, Mystery",
        "Romance":      "Romance, Drama, Comedy",
        "Horror":       "Horror, Thriller, Mystery",
        "Mystery":      "Mystery, Crime, Thriller",
        "Thriller":     "Thriller, Crime, Mystery",
        "Sci-Fi":       "Science Fiction, Fantasy, Drama",
        "Fantasy":      "Fantasy, Adventure, Animation",
        "Documentary":  "Documentary, Drama, History",
        "Animation":    "Animation, Fantasy, Comedy",
        "Other":        "Drama, Thriller",
    }
    genre_str = genre_map.get(prefs.genre, prefs.genre)

    # Overview: natural-language description of what the user wants
    overview = (
        f"Looking for a {prefs.genre} movie that is {goal_kw.split(',')[0].strip()}. "
        f"Loved {prefs.favorite_movie} because: {prefs.favorite_why[:120]}. "
        f"Recently watched {prefs.recently_watched}."
    )

    return (
        f"Title: {prefs.favorite_movie}\n"
        f"Genres: {genre_str}\n"
        f"Keywords: {all_keywords}\n"
        f"Overview: {overview}"
    )


def build_comfort_embed_chunk(result) -> str:
    """
    Build the comfort-zone user embed chunk using adjacent/expanded genres.
    Same 4-field format, but Genres field shifted outward to widen retrieval.
    """
    prefs = result.movie_prefs
    mbti  = result.mbti

    adjacent_map = {
        "Action":       "Thriller, Crime, Drama",
        "Adventure":    "Drama, Romance, History",
        "Comedy":       "Drama, Romance, Music",
        "Drama":        "History, Biography, War",
        "Romance":      "Drama, Music, Animation",
        "Horror":       "Psychological Thriller, Crime, Drama",
        "Mystery":      "Drama, History, Thriller",
        "Thriller":     "Drama, Crime, War",
        "Sci-Fi":       "Fantasy, Drama, Philosophy",
        "Fantasy":      "Drama, History, Music",
        "Documentary":  "History, War, Biography",
        "Animation":    "Family, Music, Drama",
        "Other":        "Drama, Documentary, History",
    }
    adjacent = adjacent_map.get(prefs.genre, "Drama, Documentary")

    sn_keywords = (
        "imagination, surreal, symbolic, ideas, philosophical"
        if mbti.sn == "N" else
        "realistic, grounded, practical, true story"
    )
    tf_keywords = (
        "cerebral, intellectual, clever writing"
        if mbti.tf == "T" else
        "emotional depth, character-driven, relationships"
    )
    goal_keyword_map = {
        "Funny / Light-hearted":          "comedy, humor, feel-good",
        "Emotional / Moving":             "emotional, drama, moving",
        "Suspenseful / Tense":            "tension, mystery, psychological",
        "Thought-provoking":              "thought-provoking, philosophical, identity",
        "Relaxing / Comforting":          "comfort, warmth, slice of life",
        "Educational":                    "history, true story, informative",
        "Unique / Unconventional":        "experimental, unconventional, arthouse",
        "Featuring a specific actor or director": "acclaimed, auteur",
        "Other":                          "varied",
    }
    goal_kw = goal_keyword_map.get(prefs.next_rec_goal, "varied")

    all_keywords = ", ".join(filter(None, [
        sn_keywords, tf_keywords, goal_kw,
        f"themes similar to {prefs.favorite_movie}",
    ]))

    overview = (
        f"A film outside {prefs.genre} that preserves the qualities of "
        f"{prefs.favorite_movie}: {prefs.favorite_why[:100]}. "
        f"Different genre but emotionally and tonally compatible."
    )

    return (
        f"Title: {prefs.favorite_movie}\n"
        f"Genres: {adjacent}\n"
        f"Keywords: {all_keywords}\n"
        f"Overview: {overview}"
    )


# ── Embedding cache ───────────────────────────────────────────────────────────

def _load_chunks() -> pd.DataFrame:
    """Load CSV and build focused 4-field embed_text column for each movie."""
    df = pd.read_csv(CHUNKS_CSV)
    df["movie_id"]     = pd.to_numeric(df["movie_id"],     errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0.0)
    df["popularity"]   = pd.to_numeric(df.get("popularity",    0), errors="coerce").fillna(0.0)
    df["chunk_text"]   = df["chunk_text"].fillna("").astype(str)
    df["title"]        = df["title"].fillna("").astype(str)

    # Build the 4-field embed text (Title + Genres + Keywords + Overview)
    df["embed_text"] = df.apply(
        lambda row: build_movie_embed_chunk(row["chunk_text"], row["title"]),
        axis=1
    )
    return df


def load_or_build_embeddings(verbose: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return (matrix, df) where matrix[i] is the 384-dim embedding for df.iloc[i].
    Embeddings are built from the 4-field embed_text column.
    Cached after first run — loads in ~0.1s on subsequent calls.
    """
    df = _load_chunks()

    if EMBED_CACHE.exists() and INDEX_CACHE.exists():
        if verbose:
            print("  Loading movie embeddings from cache...")
        matrix = np.load(str(EMBED_CACHE))
        with open(INDEX_CACHE) as f:
            cached_ids = json.load(f)
        current_ids = df["movie_id"].tolist()
        if cached_ids == current_ids and matrix.shape[0] == len(df):
            if verbose:
                print(f"  Cache valid: {matrix.shape[0]} movies × {matrix.shape[1]} dims")
            return matrix, df
        if verbose:
            print("  Cache mismatch — rebuilding...")

    texts = df["embed_text"].tolist()
    if verbose:
        print(f"  Building embeddings for {len(texts)} movies with {EMBED_MODEL}...")
        print("  Fields: Title + Genres + Keywords + Overview")
        print("  (~1–2 min on first run, then cached permanently)\n")
        print(f"  Sample chunk for '{df.iloc[0]['title']}':")
        for line in texts[0].splitlines():
            print(f"    {line[:90]}")
        print()

    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    matrix = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → dot product == cosine sim
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f}s — shape: {matrix.shape}")

    np.save(str(EMBED_CACHE), matrix)
    with open(INDEX_CACHE, "w") as f:
        json.dump(df["movie_id"].tolist(), f)
    if verbose:
        print(f"  Embeddings cached to {EMBED_CACHE.name}")

    return matrix, df


# ── Cosine similarity search ──────────────────────────────────────────────────

def _embed_text(text: str) -> np.ndarray:
    """Embed a single text string; returns a unit-norm 384-dim vector."""
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


def _cosine_search(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    df: pd.DataFrame,
    top_k: int,
    exclude_ids: set[int] | None = None,
) -> list[MovieHit]:
    """
    Dot product over unit-norm matrix == cosine similarity.
    Returns top_k MovieHit objects in descending similarity order.
    """
    scores = matrix @ query_vec
    ranked = np.argsort(scores)[::-1]

    hits: list[MovieHit] = []
    for idx in ranked:
        row = df.iloc[idx]
        mid = int(row["movie_id"])
        if exclude_ids and mid in exclude_ids:
            continue
        chunk = row["chunk_text"]
        hits.append(MovieHit(
            title        = row["title"],
            genres       = _parse_field(chunk, "Genres"),
            overview     = _parse_field(chunk, "Overview"),
            similarity   = float(scores[idx]),
            chunk_text   = chunk,
            movie_id     = mid,
            vote_average = float(row.get("vote_average", 0)),
            popularity   = float(row.get("popularity",   0)),
        ))
        if len(hits) >= top_k:
            break

    return hits


# ── Main entry point ──────────────────────────────────────────────────────────

def get_recommendations(result, verbose: bool = True) -> RecommendationSet:
    """
    Full pipeline:
      1. Load/cache movie embeddings (Title + Genres + Keywords + Overview)
      2. Build user profile embed chunks using the same 4-field format
      3. Embed user profile → cosine similarity against all movies
      4. Return top-5 best-fit and top-5 comfort-zone hits

    Parameters
    ----------
    result  : QuizResult from quiz.py
    verbose : print progress messages
    """
    # 1. Movie embeddings
    matrix, df = load_or_build_embeddings(verbose=verbose)

    # 2. User profile embed chunks (same 4-field format as movies)
    best_chunk    = build_user_embed_chunk(result)
    comfort_chunk = build_comfort_embed_chunk(result)

    if verbose:
        print("\n  User profile embed chunk (best-fit):")
        for line in best_chunk.splitlines():
            print(f"    {line[:100]}")
        print("\n  User profile embed chunk (comfort-zone):")
        for line in comfort_chunk.splitlines():
            print(f"    {line[:100]}")

    # 3. Embed user profile
    if verbose:
        print("\n  Embedding user profile...")
    best_vec    = _embed_text(best_chunk)
    comfort_vec = _embed_text(comfort_chunk)

    # 4. Best-fit: top-K by cosine similarity
    if verbose:
        print(f"  Searching {len(df)} movies for best-fit matches...")
    best_hits = _cosine_search(best_vec, matrix, df, top_k=TOP_K)
    best_ids  = {h.movie_id for h in best_hits}

    # 5. Comfort-zone: broader pool excluding best-fit
    if verbose:
        print("  Searching for comfort-zone recommendations...")
    comfort_pool = _cosine_search(comfort_vec, matrix, df, top_k=COMFORT_POOL)

    comfort_hits: list[MovieHit] = []
    seen = {h.title for h in best_hits}
    for candidate in comfort_pool:
        if candidate.title in seen or candidate.movie_id in best_ids:
            continue
        comfort_hits.append(candidate)
        seen.add(candidate.title)
        if len(comfort_hits) >= TOP_K:
            break

    return RecommendationSet(
        best_fit           = best_hits,
        comfort_zone       = comfort_hits,
        user_query_best    = best_chunk,
        user_query_comfort = comfort_chunk,
        embed_dim          = matrix.shape[1],
    )
