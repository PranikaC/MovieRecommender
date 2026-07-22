"""
Personality + Movie Recommendation Quiz
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

# Reuse the Bedrock client from personality_test
from personality_test import call_claude, BEARER_TOKEN

# Phase 2 — RAG recommendation engine
from recommender import get_recommendations, RecommendationSet
from rag_response import generate_rag_response, display_recommendations, RAGOutput

load_dotenv()

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MoviePreferences:
    genre: str = ""
    recently_watched: str = ""
    favorite_movie: str = ""
    favorite_why: str = ""
    next_rec_goal: str = ""

@dataclass
class PersonalityScores:
    """Raw Likert scores — never shown to user."""
    q5_social: int = 0       # E indicator
    q6_openness: int = 0     # N indicator
    q7_logic: int = 0        # T indicator
    q8_planning: int = 0     # J indicator
    q9_solitude: int = 0     # I indicator
    q10_patterns: int = 0    # N indicator
    q11_empathy: int = 0     # F indicator
    q12_organized: int = 0   # J indicator

@dataclass
class MBTIResult:
    """Deterministic MBTI-style result — never shown to user as raw data."""
    ei: str = ""             # "E", "I", or "Balanced"
    sn: str = ""             # "S" or "N"
    tf: str = ""             # "T", "F", or "Balanced"
    jp: str = ""             # "J" or "P"
    at: str = ""             # "A", "T", or "uncertain"
    ei_balanced: bool = False
    tf_balanced: bool = False
    full_type: str = ""      # e.g. "ENFP"
    type_name: str = ""      # e.g. "The Campaigner"

@dataclass
class QuizResult:
    movie_prefs: MoviePreferences = field(default_factory=MoviePreferences)
    scores: PersonalityScores = field(default_factory=PersonalityScores)
    mbti: MBTIResult = field(default_factory=MBTIResult)
    profile_text: str = ""
    recs: Optional[RecommendationSet] = None
    rag_output: Optional[RAGOutput] = None

# ── MBTI type name lookup ─────────────────────────────────────────────────────

TYPE_NAMES = {
    "INTJ": "The Architect",
    "INTP": "The Logician",
    "ENTJ": "The Commander",
    "ENTP": "The Debater",
    "INFJ": "The Advocate",
    "INFP": "The Mediator",
    "ENFJ": "The Protagonist",
    "ENFP": "The Campaigner",
    "ISTJ": "The Logistician",
    "ISFJ": "The Defender",
    "ESTJ": "The Executive",
    "ESFJ": "The Consul",
    "ISTP": "The Virtuoso",
    "ISFP": "The Adventurer",
    "ESTP": "The Entrepreneur",
    "ESFP": "The Entertainer",
}

# ── Quiz questions ────────────────────────────────────────────────────────────

GENRE_OPTIONS = [
    "Action", "Adventure", "Comedy", "Drama", "Romance",
    "Horror", "Mystery", "Thriller", "Sci-Fi", "Fantasy",
    "Documentary", "Animation", "Other"
]

REC_GOAL_OPTIONS = [
    "Funny / Light-hearted",
    "Emotional / Moving",
    "Suspenseful / Tense",
    "Thought-provoking",
    "Relaxing / Comforting",
    "Educational",
    "Unique / Unconventional",
    "Featuring a specific actor or director",
    "Other",
]

LIKERT_LABEL = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly Agree",
}

LIKERT_STATEMENTS = {
    5:  "I enjoy meeting and talking with new people.",
    6:  "I like trying new ideas, experiences, or hobbies.",
    7:  "I usually make decisions based on logic rather than emotions.",
    8:  "I prefer having a plan instead of figuring things out as I go.",
    9:  "I enjoy spending time by myself to recharge.",
    10: "I often notice patterns or imagine possibilities that others might overlook.",
    11: "When making decisions, I consider how my choices affect other people.",
    12: "I like to keep my schedule organized and finish tasks before deadlines.",
}

# ── Display helpers ───────────────────────────────────────────────────────────

def divider(char="─", width=62):
    print(char * width)

def header(title: str):
    divider("═")
    print(f"  {title}")
    divider("═")

def prompt_choice(options: list[str], allow_other: bool = False) -> str:
    """Display a numbered list and return the chosen value."""
    for i, opt in enumerate(options, 1):
        print(f"  {i:>2}. {opt}")
    while True:
        raw = input("\nEnter number: ").strip()
        if raw.lower() == "quit":
            raise SystemExit("\nQuiz exited. Goodbye!")
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")

def prompt_text(prompt: str, required: bool = True) -> str:
    """Prompt for free-text input."""
    while True:
        val = input(f"{prompt} ").strip()
        if val.lower() == "quit":
            raise SystemExit("\nQuiz exited. Goodbye!")
        if val or not required:
            return val
        print("  (Please enter a response to continue.)")

def prompt_likert(statement: str) -> int:
    """Display a Likert scale and return 1–5."""
    print(f"\n  \"{statement}\"")
    print()
    for k, v in LIKERT_LABEL.items():
        print(f"    {k} — {v}")
    while True:
        raw = input("\n  Your rating (1–5): ").strip()
        if raw.lower() == "quit":
            raise SystemExit("\nQuiz exited. Goodbye!")
        try:
            val = int(raw)
            if 1 <= val <= 5:
                return val
        except ValueError:
            pass
        print("  Please enter a number from 1 to 5.")

# ── Quiz flow ─────────────────────────────────────────────────────────────────

def run_movie_questions() -> MoviePreferences:
    prefs = MoviePreferences()

    print("\n📽️  Part 1 of 2 — Your Movie Taste\n")
    divider()

    # Q1 — Genre
    print("\nQuestion 1 of 12")
    print("What's your favorite movie genre?\n")
    prefs.genre = prompt_choice(GENRE_OPTIONS)

    # Q2 — Recently watched
    print("\nQuestion 2 of 12")
    prefs.recently_watched = prompt_text(
        "What's the last movie (or two) you watched?"
    )

    # Q3 — Favorite movie + why
    print("\nQuestion 3 of 12")
    prefs.favorite_movie = prompt_text("What's your all-time favorite movie?")
    prefs.favorite_why = prompt_text(
        f"  What do you love most about \"{prefs.favorite_movie}\"?"
    )

    # Q4 — Next rec goal
    print("\nQuestion 4 of 12")
    print("What are you looking for in your next movie recommendation?\n")
    prefs.next_rec_goal = prompt_choice(REC_GOAL_OPTIONS)

    return prefs


def run_personality_questions() -> PersonalityScores:
    scores = PersonalityScores()

    print("\n\n🧠  Part 2 of 2 — A Bit About You")
    print("  Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree).\n")
    divider()

    q_map = {
        5:  "q5_social",
        6:  "q6_openness",
        7:  "q7_logic",
        8:  "q8_planning",
        9:  "q9_solitude",
        10: "q10_patterns",
        11: "q11_empathy",
        12: "q12_organized",
    }

    for q_num, attr in q_map.items():
        print(f"\nQuestion {q_num} of 12")
        val = prompt_likert(LIKERT_STATEMENTS[q_num])
        setattr(scores, attr, val)

    return scores


# ── Scoring ───────────────────────────────────────────────────────────────────

def calculate_mbti(scores: PersonalityScores) -> MBTIResult:
    result = MBTIResult()

    # E / I
    if scores.q5_social > scores.q9_solitude:
        result.ei = "E"
    elif scores.q9_solitude > scores.q5_social:
        result.ei = "I"
    else:
        result.ei = "E"           # tie-break: default to E for label
        result.ei_balanced = True

    # S / N  (average of Q6 + Q10)
    n_avg = (scores.q6_openness + scores.q10_patterns) / 2
    result.sn = "N" if n_avg >= 3.5 else "S"

    # T / F
    if scores.q7_logic > scores.q11_empathy:
        result.tf = "T"
    elif scores.q11_empathy > scores.q7_logic:
        result.tf = "F"
    else:
        result.tf = "T"           # tie-break: default to T for label
        result.tf_balanced = True

    # J / P  (average of Q8 + Q12)
    j_avg = (scores.q8_planning + scores.q12_organized) / 2
    result.jp = "J" if j_avg >= 3.5 else "P"

    # Full type
    result.full_type = result.ei + result.sn + result.tf + result.jp
    result.type_name = TYPE_NAMES.get(result.full_type, "The Individual")

    # Assertive / Turbulent — heuristic from scores
    # High logic + high planning + low empathy weighting → Assertive
    # High empathy + low planning + lower logic → Turbulent
    confidence_score = (scores.q7_logic + scores.q8_planning + scores.q12_organized) / 3
    sensitivity_score = (scores.q11_empathy + (6 - scores.q8_planning)) / 2  # inverse planning
    if confidence_score >= 3.8 and sensitivity_score < 3.5:
        result.at = "A"
    elif sensitivity_score >= 3.8 or (scores.q11_empathy >= 4 and scores.q8_planning <= 2):
        result.at = "T"
    else:
        result.at = "uncertain"

    return result


# ── Profile generation ────────────────────────────────────────────────────────

SYSTEM_PROFILE = """You are a warm, insightful personality and movie analyst. Your job is to generate 
a friendly, fun, and accurate personality + movie profile based on structured quiz data.

STRICT RULES — follow these exactly:
1. NEVER display raw scores, numbers, question numbers, or internal calculations.
2. NEVER mention the quiz structure, question labels, or Likert scale.
3. NEVER make clinical or medical claims. Always frame this as entertainment.
4. Write in second person ("you are", "you tend to", "you might enjoy").
5. Be warm, specific, and grounded in the data provided. Avoid vague horoscope language.
6. Keep the entire response under 500 tokens.

OUTPUT FORMAT — use exactly these sections with these headers:

## [MBTI Type Code] — [Type Name]
*[one-line tagline for this person]*

### Personality Summary
[2 short paragraphs about how they approach life, decisions, people, and entertainment]

### Your Traits at a Glance
- **Social energy:** [Extraverted / Introverted / Balanced]
- **Imagination style:** [Practical / Imaginative / Balanced]  
- **Decision style:** [Logic-led / Values-led / Balanced]
- **Lifestyle style:** [Structured / Flexible / Balanced]

### Your Movie Taste
[1 short paragraph summarizing their genre preference, favorite movie, recently watched, and rec goal]

### What to Watch Next
[2–4 bullet points describing the *types* of movies that would suit them — no specific titles needed]

### Why This Profile?
[1–2 sentences explaining what quiz signals led to this result. Keep it light.]"""


def build_profile_prompt(result: QuizResult) -> str:
    mbti = result.mbti
    scores = result.scores
    prefs = result.movie_prefs

    # Trait labels for the prompt
    ei_label = "Balanced (slight lean Extraverted)" if mbti.ei_balanced else (
        "Extraverted" if mbti.ei == "E" else "Introverted"
    )
    tf_label = "Balanced (slight lean Logic-led)" if mbti.tf_balanced else (
        "Logic-led" if mbti.tf == "T" else "Values-led"
    )
    sn_label = "Imaginative" if mbti.sn == "N" else "Practical"
    jp_label = "Structured" if mbti.jp == "J" else "Flexible"
    at_label = mbti.at if mbti.at != "uncertain" else "unclear"

    # Describe Likert patterns in natural language (never show raw numbers)
    likert_desc = []
    if scores.q5_social >= 4:
        likert_desc.append("strongly drawn to social interaction")
    elif scores.q5_social <= 2:
        likert_desc.append("tends to find large social settings draining")
    if scores.q9_solitude >= 4:
        likert_desc.append("recharges best in solitude")
    elif scores.q9_solitude <= 2:
        likert_desc.append("rarely needs alone time to recharge")
    if scores.q6_openness >= 4 or scores.q10_patterns >= 4:
        likert_desc.append("genuinely curious and open to new ideas")
    if scores.q7_logic >= 4:
        likert_desc.append("favors logical, analytical thinking")
    if scores.q11_empathy >= 4:
        likert_desc.append("highly attuned to how decisions affect others")
    if scores.q8_planning >= 4 or scores.q12_organized >= 4:
        likert_desc.append("prefers structure and planning ahead")
    elif scores.q8_planning <= 2 and scores.q12_organized <= 2:
        likert_desc.append("comfortable with spontaneity and loose plans")

    personality_signals = "; ".join(likert_desc) if likert_desc else "moderate across all personality dimensions"

    return f"""Please generate a personality + movie profile for the following person.

PERSONALITY DATA (internal — do not display):
- Estimated MBTI-style type: {mbti.full_type} — {mbti.type_name}
- E/I axis: {ei_label}
- S/N axis: {sn_label}
- T/F axis: {tf_label}
- J/P axis: {jp_label}
- Assertive/Turbulent: {at_label}
- Personality signals: {personality_signals}

MOVIE PREFERENCES:
- Favorite genre: {prefs.genre}
- Recently watched: {prefs.recently_watched}
- Favorite movie: {prefs.favorite_movie}
- Why they love it: {prefs.favorite_why}
- Looking for next: {prefs.next_rec_goal}

Generate the profile now following the required output format exactly."""


def generate_profile(result: QuizResult) -> str:
    prompt = build_profile_prompt(result)
    return call_claude(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=SYSTEM_PROFILE,
        max_tokens=500,
        timeout=60,
    )


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(result: QuizResult):
    save = input("\nSave your profile and recommendations to a file? (y/n): ").strip().lower()
    if save != "y":
        return

    filename = "quiz_results.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NETFLIX RECOMMENDER — PERSONALITY + MOVIE PROFILE\n")
        f.write("=" * 62 + "\n\n")
        f.write(result.profile_text)

        if result.rag_output:
            out = result.rag_output
            f.write("\n\n" + "=" * 62 + "\n")
            f.write("RECOMMENDED FOR YOU\n")
            f.write("=" * 62 + "\n\n")
            for i, rec in enumerate(out.best_fit, 1):
                f.write(f"{i}. {rec.title}\n")
                f.write(f"   {rec.reason}\n")
                f.write(f"   Genre/Tone: {rec.genre_tone}\n\n")

            f.write("\nTRY SOMETHING DIFFERENT\n")
            f.write("-" * 62 + "\n\n")
            for i, rec in enumerate(out.comfort_zone, 1):
                f.write(f"{i}. {rec.title}\n")
                f.write(f"   {rec.reason}\n")
                if rec.stretch_note:
                    f.write(f"   {rec.stretch_note}\n")
                f.write(f"   Genre/Tone: {rec.genre_tone}\n\n")

            if out.why_paragraph:
                f.write("\nWhy these fit your profile:\n")
                f.write(out.why_paragraph + "\n")

        f.write("\n" + "=" * 62 + "\n")
        f.write("Generated by Netflix Recommender · Claude Sonnet 4-6\n")
    print(f"  Saved to {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def print_banner():
    print()
    divider("═")
    print("   🎬  NETFLIX RECOMMENDER — PERSONALITY QUIZ  🎬")
    print("   Powered by Claude Sonnet 4-6 · AWS Bedrock")
    divider("═")
    print("""
  This short quiz helps us understand your taste and personality
  so we can point you toward movies you'll actually love.

  12 quick questions — takes about 2 minutes.
  Type 'quit' at any time to exit.
  This is for fun — not a clinical assessment!
""")


def main():
    if not BEARER_TOKEN:
        print("\nERROR: AWS_BEARER_TOKEN_BEDROCK is not set in your .env file.")
        print("Add your Bedrock API key and try again.\n")
        return

    print_banner()

    input("  Press Enter to start the quiz...")

    try:
        # ── Collect answers ──
        movie_prefs = run_movie_questions()
        scores      = run_personality_questions()

        # ── Score internally ──
        mbti = calculate_mbti(scores)

        result = QuizResult(
            movie_prefs=movie_prefs,
            scores=scores,
            mbti=mbti,
        )

        # ── Generate profile ──
        print("\n\n" + "─" * 62)
        print("  ✨ Building your profile — just a moment...")
        print("─" * 62)

        result.profile_text = generate_profile(result)

        # ── Display profile ──
        print("\n")
        header("🎬  YOUR PERSONALITY + MOVIE PROFILE")
        print()
        print(result.profile_text)
        print()
        divider("═")

        # ── Phase 2: Recommendations ──
        print("\n" + "─" * 62)
        print("  🔍 Finding your movie matches...")
        print("─" * 62)

        result.recs = get_recommendations(result, verbose=True)
        result.rag_output = generate_rag_response(result.recs, result, verbose=True)

        display_recommendations(result.rag_output)

        save_results(result)

    except SystemExit as e:
        print(str(e))
    except RuntimeError as e:
        print(f"\nAPI Error: {e}")
    except KeyboardInterrupt:
        print("\n\nQuiz interrupted. Goodbye!")


if __name__ == "__main__":
    main()
