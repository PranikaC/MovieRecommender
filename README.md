# Netflix Recommender — Personality-Based Movie Recommendations

A prototype recommendation system that combines a short personality quiz with vector embedding search to deliver personalized movie recommendations from a dataset of 4,799 films.

## How It Works

```
12-Question Quiz
    ├── Questions 1–4:  Movie preferences (genre, favorites, recently watched, goals)
    └── Questions 5–12: Personality traits (Likert scale 1–5)
            │
            ▼
    Deterministic MBTI-Style Scoring (E/I, S/N, T/F, J/P)
            │
            ▼
    Claude Sonnet 4 generates a personality + movie profile
            │
            ▼
    User profile → embedded as a 4-field vector (Title, Genres, Keywords, Overview)
    Movie dataset → embedded with the same 4-field structure
            │
            ▼
    Cosine similarity search across 4,799 movies
            │
            ▼
    Top 5 Best-Fit + Top 5 Outside-Comfort-Zone Recommendations
```

## Quick Start

### Prerequisites

- Python 3.11+
- AWS credentials with Bedrock access (bearer token)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/NetflixRecommender.git
cd NetflixRecommender

# Create virtual environment
python -m venv venv

# Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Credentials

Create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_SESSION_TOKEN=your_token_here
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_DEFAULT_REGION=us-west-2
```

### Run the Quiz

```bash
python quiz.py
```

The first run takes ~2 minutes to build the movie embedding cache. Every subsequent run loads in under a second.

## Project Structure

```
NetflixRecommender/
├── quiz.py                  # Main entry point — 12-question quiz + recommendations
├── personality_test.py      # Bedrock API client (call_claude) + standalone interview mode
├── recommender.py           # Embedding engine — builds/caches vectors, cosine search
├── rag_response.py          # Formats recommendations (Claude RAG or deterministic fallback)
├── activate.ps1             # PowerShell convenience script (loads .env + activates venv)
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes .env, venv/, cache files
├── data/
│   ├── final_movie_chunks_for_rag.csv   # 4,799-movie dataset
│   ├── movie_embeddings.npy             # Cached embeddings (auto-generated)
│   └── movie_titles_index.json          # Cache index (auto-generated)
├── EmbeddingAssignment.ipynb            # Notebook: embedding model exploration
└── NetflixRecommenderChunks.ipynb       # Notebook: TMDB data → chunk CSV pipeline
```

## Technical Details

### Embedding Strategy

Both movie chunks and user profiles use the same 4-field format for embedding:

```
Title: <title>
Genres: <genres>
Keywords: <keywords>
Overview: <overview>
```

Director, cast, and tagline are intentionally excluded — they add noise without contributing to genre/tone/theme matching. This ensures cosine similarity operates in a consistent semantic space.

- **Model:** `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Similarity:** Cosine similarity via normalized dot product
- **Dataset:** 4,799 movies from TMDB with rich metadata

### Personality Scoring

MBTI-style type is calculated deterministically from Likert responses:
- **E/I:** Social energy (Q5 vs Q9)
- **S/N:** Imagination style (average of Q6 + Q10)
- **T/F:** Decision style (Q7 vs Q11)
- **J/P:** Lifestyle structure (average of Q8 + Q12)

The LLM is only used for generating the written profile and formatting recommendations — never for scoring.

### Recommendation Pipeline

1. **Best-fit:** User profile embedding compared against all 4,799 movie embeddings. Top 5 returned.
2. **Comfort-zone expansion:** A second query with shifted genres retrieves candidates from adjacent territory. Top 5 returned (no overlap with best-fit).
3. **RAG grounding:** If Claude is available, it writes personalized explanations grounded only in the retrieved movies. If unavailable, a deterministic fallback formats the results directly.

## LLM

- **Model:** Claude Sonnet 4 (`global.anthropic.claude-sonnet-4-6`) via AWS Bedrock
- **Auth:** Bearer token (Bedrock API key)
- The system works without the LLM — the deterministic fallback provides recommendations using the retrieved dataset results directly.

## Disclaimer

This is an entertainment-style personality quiz, not a formal psychological assessment. MBTI-style types are approximations for fun and should not be interpreted as clinical diagnoses.
