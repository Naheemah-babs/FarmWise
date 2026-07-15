"""
FarmWise crew: 3 collaborating CrewAI agents.

  1. Crop Advisor      - interprets the farmer's description, identifies
                          likely issues from the local agronomy guide.
  2. Market Analyst     - advises on selling timing and fair price range
                          using the local market price table.
  3. Action Recommender - gives a clear next step; escalates suspected
                          serious disease/pest outbreaks to a human officer.

The crew is treated as an UNTRUSTED reasoning engine. Guardrails in
guardrails.py run before and after this crew is called — this file only
defines the agents/tasks, it does not decide what's safe to show a user.
"""

import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

from guardrails import load_agronomy_guide, load_market_prices

load_dotenv()

# ---------------------------------------------------------------------------
# LLM configuration — OpenRouter's free auto-router, via LiteLLM's
# "openrouter/" provider prefix (which CrewAI uses under the hood).
#
# "openrouter/openrouter/free" is OpenRouter's own $0 router: it picks a
# free-tier model per request based on what the request needs, so it keeps
# working even as individual free models rotate in/out. No credit card
# needed — just an OpenRouter account and API key.
#
# Free-tier models are rate-limited (roughly 20 requests/minute, 200/day),
# which is fine for a course project/demo but not for production traffic.
#
# Swap models/providers freely via env vars — no code change needed:
#   FARMWISE_MODEL       e.g. "openrouter/openrouter/free" (default)
#   OPENROUTER_API_KEY   your OpenRouter API key (openrouter.ai/keys)
#
# To pin a specific free model instead of the auto-router, set e.g.:
#   FARMWISE_MODEL=openrouter/meta-llama/llama-3.3-70b-instruct:free
# ---------------------------------------------------------------------------

FARMWISE_MODEL = os.getenv("FARMWISE_MODEL", "openrouter/openrouter/free")

llm = LLM(
    model=FARMWISE_MODEL,
    temperature=0.3,
)


def build_crew(crop: str, location: str, problem_description: str) -> Crew:
    agronomy_guide = load_agronomy_guide()
    market_prices = load_market_prices()

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    crop_advisor = Agent(
        role="Crop Advisor",
        goal=(
            "Diagnose the farmer's crop problem using ONLY the provided "
            "local agronomy guide, and clearly state whether the issue "
            "should be marked SEVERE / ESCALATE."
        ),
        backstory=(
            "You are a field-experienced agronomist who has spent years "
            "walking smallholder farms in West Africa. You know that "
            "guessing at a diagnosis can cost a farmer their harvest, so "
            "you strictly stick to the local agronomy guide you've been "
            "given rather than inventing treatments. When something doesn't "
            "match the guide clearly, you say so plainly instead of bluffing."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    market_analyst = Agent(
        role="Market Analyst",
        goal=(
            "Advise the farmer on fair price expectations and selling "
            "timing using ONLY the provided local market price table."
        ),
        backstory=(
            "You are a market analyst who tracks local crop prices for "
            "smallholder cooperatives. You never quote a price outside the "
            "provided table, and you always mention the seasonal pattern "
            "so farmers know whether to sell now or wait."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    action_recommender = Agent(
        role="Action Recommender",
        goal=(
            "Turn the Crop Advisor's diagnosis and the Market Analyst's "
            "guidance into ONE clear, short next step for the farmer. "
            "If the diagnosis is marked SEVERE / ESCALATE, or describes a "
            "widespread/fast-spreading problem, the recommended action "
            "MUST be to contact a human agricultural extension officer — "
            "never a home treatment."
        ),
        backstory=(
            "You are the practical voice that a busy smallholder farmer "
            "actually reads. You cut through detail to give one plain, "
            "actionable instruction, and you never soften or skip an "
            "escalation when one is warranted — a farmer's livelihood is "
            "on the line."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ------------------------------------------------------------------
    # Tasks (sequential — each builds on the last, like SafeSpend)
    # ------------------------------------------------------------------

    crop_task = Task(
        description=(
            f"A farmer growing {crop} in {location} reports this problem:\n"
            f'"{problem_description}"\n\n'
            f"Local agronomy guide (your ONLY source of truth):\n"
            f"---\n{agronomy_guide}\n---\n\n"
            "Identify the most likely issue from the guide, cite the guide "
            "entry by name, and state clearly whether it is marked "
            "SEVERE / ESCALATE or safe for the farmer to act on directly. "
            "If nothing in the guide clearly matches, say so explicitly "
            "instead of guessing."
        ),
        expected_output=(
            "2-4 sentences: likely issue (naming the guide entry), whether "
            "it is SEVERE / ESCALATE, and the guide's recommended action."
        ),
        agent=crop_advisor,
    )

    market_task = Task(
        description=(
            f"The farmer grows {crop} and is located in {location}.\n\n"
            f"Local market price table (your ONLY source of truth):\n"
            f"{market_prices}\n\n"
            "Give a short note on the fair price range and the best timing "
            "to sell this crop, based only on this table. If the crop is "
            "not in the table, say clearly that no local price data is "
            "available rather than estimating one."
        ),
        expected_output="1-3 sentences on price range and selling timing.",
        agent=market_analyst,
    )

    action_task = Task(
        description=(
            "Using the Crop Advisor's diagnosis and the Market Analyst's "
            "note, write ONE clear recommended next step for the farmer. "
            "If the diagnosis was marked SEVERE / ESCALATE, or the farmer's "
            "own words describe a widespread/fast-spreading problem, the "
            "action MUST be to contact a human agricultural extension "
            "officer immediately — do not suggest a home treatment in that "
            "case."
        ),
        expected_output=(
            "A JSON object with exactly these keys: "
            '"crop_advice", "market_advice", "recommended_action", '
            '"escalate" (true/false). No extra commentary outside the JSON.'
        ),
        agent=action_recommender,
        context=[crop_task, market_task],
    )

    crew = Crew(
        agents=[crop_advisor, market_analyst, action_recommender],
        tasks=[crop_task, market_task, action_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew
