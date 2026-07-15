"""
FarmWise guardrails.

Implements 4 of the Week 9 guardrails, wrapped around the CrewAI crew
(the crew is treated as an untrusted reasoning engine):

  1. Input validation   -> validate_input()
  2. Grounding           -> check_grounding()
  3. Human-in-the-loop   -> check_escalation()
  4. Logging             -> log_interaction()

These run OUTSIDE the LLM, in plain Python, so they can't be talked out
of doing their job by the crew's output.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "farmwise_queries.jsonl"

AGRONOMY_GUIDE_PATH = DATA_DIR / "agronomy_guide.md"
MARKET_PRICES_PATH = DATA_DIR / "market_prices.json"

# ---------------------------------------------------------------------------
# Shared local knowledge loaders (also used by crew.py so agents are grounded
# in the same files the guardrails check against)
# ---------------------------------------------------------------------------

def load_agronomy_guide() -> str:
    return AGRONOMY_GUIDE_PATH.read_text(encoding="utf-8")


def load_market_prices() -> dict:
    return json.loads(MARKET_PRICES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. INPUT VALIDATION
# ---------------------------------------------------------------------------

# Known crops we actually have agronomy data for.
KNOWN_CROPS = {"maize", "cassava", "tomato", "cowpea", "beans"}

# Crude prompt-injection / malicious-input patterns to block outright.
BLOCKED_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (all|any|the) (previous|prior|above) (instructions|rules)",
    r"system prompt",
    r"you are now",
    r"act as (an?|the) (?!farmer|agronomist|market analyst)",
    r"reveal your (instructions|prompt|guardrails)",
    r"<script",
    r"drop table",
    r"rm -rf",
]


def validate_input(crop: str, location: str, problem_description: str) -> dict:
    """
    Returns a dict:
      {"ok": True, "crop": <normalized>, "notes": [...]}
    or
      {"ok": False, "reason": <str>, "blocked": <bool>}

    Handles vague/incomplete descriptions gracefully (asks for more info
    rather than crashing or guessing), and blocks obviously malicious input.
    """
    crop = (crop or "").strip().lower()
    location = (location or "").strip()
    problem_description = (problem_description or "").strip()

    # --- malicious input check (checked first, across all fields) ---
    combined = f"{crop} {location} {problem_description}".lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return {
                "ok": False,
                "blocked": True,
                "reason": (
                    "Your message could not be processed because it looks like "
                    "an attempt to manipulate the system rather than a genuine "
                    "farming question. Please rephrase as a normal question "
                    "about a crop problem."
                ),
            }

    # --- basic length / spam sanity checks ---
    if len(problem_description) > 1500:
        return {
            "ok": False,
            "blocked": True,
            "reason": "Description is too long. Please shorten it to a few sentences.",
        }

    # --- missing/vague fields: handled gracefully, not a crash ---
    notes = []
    if not crop:
        return {
            "ok": False,
            "blocked": False,
            "reason": (
                "I need to know which crop you're asking about "
                "(e.g. maize, cassava, tomato, cowpea) before I can help."
            ),
        }

    if crop not in KNOWN_CROPS:
        notes.append(
            f"'{crop}' is not one of the crops in our local guide "
            f"({', '.join(sorted(KNOWN_CROPS))}). We'll do our best, but this "
            "may need escalation to a human officer."
        )

    if len(problem_description) < 8:
        return {
            "ok": False,
            "blocked": False,
            "reason": (
                "Could you describe the problem a bit more? For example: "
                "'my maize leaves are turning yellow starting from the tip', "
                "or 'holes appearing in my cowpea pods'."
            ),
        }

    if not location:
        notes.append("No location given — market timing advice will be general, not local.")

    return {
        "ok": True,
        "crop": crop,
        "location": location or "unspecified",
        "problem_description": problem_description,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 2. GROUNDING
# ---------------------------------------------------------------------------

def check_grounding(crop: str, advisor_output: str) -> dict:
    """
    Verifies the Crop Advisor's output is actually anchored in the agronomy
    guide rather than an invented treatment.

    Approach: pull the guide sections relevant to this crop, extract the
    known "signal" terms (disease/pest names, key phrases) for that crop,
    and confirm the advisor's answer overlaps with at least one of them.
    This is a lightweight, auditable check — not a proof of correctness,
    but it catches answers that are clearly untethered from the local guide.
    """
    guide = load_agronomy_guide()
    crop_upper = crop.upper()

    # Extract the section of the guide for this crop (between its heading
    # and the next '## ' heading).
    section_match = re.search(
        rf"## {re.escape(crop_upper)}.*?(?=\n## |\Z)", guide, flags=re.DOTALL
    )

    if not section_match:
        return {
            "grounded": False,
            "reason": f"No agronomy guide section exists for '{crop}'. Cannot ground advice — escalate instead.",
        }

    section_text = section_match.group(0)

    # Vocabulary = every significant word used anywhere in this crop's
    # section of the guide (headings, bold labels, and body text alike),
    # so a paraphrased-but-genuine answer still counts as grounded.
    STOPWORDS = {
        "likely", "cause", "signs", "action", "recommended", "escalate",
        "severe", "general", "these", "there", "which", "where", "should",
        "never", "always", "before", "after", "within", "requires",
        "guidance", "officer", "field", "plant", "plants", "leaves",
        "farmer", "extension",
    }
    words = re.findall(r"[a-z]{5,}", section_text.lower())
    word_terms = {w for w in words if w not in STOPWORDS}

    advisor_lower = advisor_output.lower()
    word_overlap = sorted({w for w in word_terms if w in advisor_lower})

    if word_overlap:
        return {
            "grounded": True,
            "matched_terms": word_overlap[:5],
        }

    return {
        "grounded": False,
        "reason": (
            "Advisor output does not clearly match any known term in the "
            "agronomy guide for this crop. Treat as ungrounded — escalate "
            "rather than act on it."
        ),
    }


# ---------------------------------------------------------------------------
# 3. HUMAN-IN-THE-LOOP (escalation)
# ---------------------------------------------------------------------------

SEVERE_MARKERS = [
    "severe / escalate",
    "severe/escalate",
    "escalate",
]

WIDESPREAD_MARKERS = [
    "whole field",
    "entire field",
    "most of the field",
    "most of the plants",
    "spreading fast",
    "spreading rapidly",
    "across the field",
    "all my plants",
]


def check_escalation(problem_description: str, advisor_output: str) -> dict:
    """
    Decides whether this query must be escalated to a human extension
    officer instead of (or in addition to) being answered directly.

    Escalates when:
      - The matched agronomy guide entry is marked SEVERE / ESCALATE, or
      - The farmer's own description signals a widespread/fast-spreading
        problem, regardless of what the guide entry says.
    """
    text = f"{problem_description} {advisor_output}".lower()

    reasons = []
    if any(m in text for m in SEVERE_MARKERS):
        reasons.append("Matched agronomy guide entry is marked SEVERE / ESCALATE.")
    if any(m in text for m in WIDESPREAD_MARKERS):
        reasons.append("Farmer describes widespread or fast-spreading symptoms.")

    return {
        "escalate": bool(reasons),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 4. LOGGING
# ---------------------------------------------------------------------------

def log_interaction(
    crop: str,
    location: str,
    problem_description: str,
    result: dict,
    blocked: bool = False,
    escalated: bool = False,
) -> str:
    """
    Appends one JSON line per query to logs/farmwise_queries.jsonl so the
    team can spot common regional problems over time. Returns the query id.
    """
    query_id = str(uuid.uuid4())
    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crop": crop,
        "location": location,
        "problem_description": problem_description,
        "blocked": blocked,
        "escalated": escalated,
        "result_summary": {
            k: result.get(k)
            for k in ("crop_advice", "market_advice", "recommended_action")
            if isinstance(result, dict)
        } if isinstance(result, dict) else str(result),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return query_id
