"""
FarmWise FastAPI wrapper.

Pattern (same shape as SafeSpend): guard the input -> run the crew ->
guard the output -> keep a human in the loop.
"""

import json
import re

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from crew import build_crew
from guardrails import (
    check_escalation,
    check_grounding,
    log_interaction,
    validate_input,
)

app = FastAPI(title="FarmWise", description="A crop and market advisor for smallholder farmers")


class AgentRunRequest(BaseModel):
    crop: str = ""
    location: str = ""
    problem_description: str = ""


def _extract_json(raw: str) -> dict:
    """The Action Recommender is asked for JSON; be defensive about parsing it."""
    raw = raw.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "crop_advice": raw,
            "market_advice": "",
            "recommended_action": (
                "Could not parse a structured recommendation — please contact "
                "an agricultural extension officer to be safe."
            ),
            "escalate": True,
        }


@app.get("/health")
def health():
    return {"status": "ok", "service": "FarmWise"}


@app.post("/agent/run")
async def agent_run(payload: AgentRunRequest):
    # ---- 1. GUARD THE INPUT ----
    validation = validate_input(payload.crop, payload.location, payload.problem_description)

    if not validation["ok"]:
        log_interaction(
            crop=payload.crop,
            location=payload.location,
            problem_description=payload.problem_description,
            result={"blocked_reason": validation["reason"]},
            blocked=validation.get("blocked", False),
            escalated=False,
        )
        status_code = 400
        return JSONResponse(
            status_code=status_code,
            content={
                "ok": False,
                "blocked": validation.get("blocked", False),
                "message": validation["reason"],
            },
        )

    crop = validation["crop"]
    location = validation["location"]
    problem_description = validation["problem_description"]

    # ---- 2. RUN THE CREW (untrusted reasoning engine) ----
    crew = build_crew(crop, location, problem_description)
    crew_result = await crew.kickoff_async()
    result = _extract_json(str(crew_result))

    # ---- 3. GUARD THE OUTPUT ----
    grounding = check_grounding(crop, result.get("crop_advice", ""))
    escalation = check_escalation(problem_description, result.get("crop_advice", ""))

    must_escalate = escalation["escalate"] or bool(result.get("escalate")) or not grounding["grounded"]

    if must_escalate:
        result["recommended_action"] = (
            "Please contact your local agricultural extension officer before "
            "taking any action. " + result.get("recommended_action", "")
        ).strip()
        result["escalate"] = True

    response = {
        "ok": True,
        "crop": crop,
        "location": location,
        "crop_advice": result.get("crop_advice", ""),
        "market_advice": result.get("market_advice", ""),
        "recommended_action": result.get("recommended_action", ""),
        "escalated": must_escalate,
        "grounding": grounding,
        "notes": validation.get("notes", []),
    }

    # ---- 4. LOG ----
    log_interaction(
        crop=crop,
        location=location,
        problem_description=problem_description,
        result=response,
        blocked=False,
        escalated=must_escalate,
    )

    return response


@app.get("/", response_class=HTMLResponse)
def form():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FarmWise</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #f4f7f2; margin: 0; padding: 16px; color: #23301e; }
  .card { max-width: 480px; margin: 0 auto; background: #fff; border-radius: 14px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  h1 { font-size: 1.4rem; color: #2f5d33; margin-bottom: 4px; }
  p.sub { color: #5c6b57; margin-top: 0; font-size: 0.9rem; }
  label { display: block; font-weight: 600; margin-top: 14px; margin-bottom: 4px; font-size: 0.9rem; }
  input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #c7d3c2; border-radius: 8px; font-size: 1rem; }
  textarea { min-height: 90px; resize: vertical; }
  button { margin-top: 18px; width: 100%; padding: 12px; background: #2f5d33; color: #fff; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; }
  button:disabled { background: #9db096; }
  #result { margin-top: 18px; padding: 14px; border-radius: 10px; background: #eef4ea; font-size: 0.95rem; white-space: pre-wrap; display: none; }
  #result.escalate { background: #fdeeee; border: 1px solid #e3a1a1; }
  .label-tag { display: inline-block; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; margin-bottom: 8px; }
  .tag-ok { background: #d7ead1; color: #2f5d33; }
  .tag-escalate { background: #f6cccc; color: #8a2b2b; }
</style>
</head>
<body>
  <div class="card">
    <h1>🌾 FarmWise</h1>
    <p class="sub">Describe your crop problem and get grounded, practical advice.</p>

    <label for="crop">Crop</label>
    <input id="crop" placeholder="e.g. maize" />

    <label for="location">Location (optional)</label>
    <input id="location" placeholder="e.g. Oyo State" />

    <label for="problem">What's happening?</label>
    <textarea id="problem" placeholder="e.g. my maize leaves are yellowing from the tip"></textarea>

    <button id="submitBtn" onclick="runAgent()">Get Advice</button>

    <div id="result"></div>
  </div>

<script>
async function runAgent() {
  const btn = document.getElementById('submitBtn');
  const resultEl = document.getElementById('result');
  btn.disabled = true;
  btn.textContent = 'Thinking...';
  resultEl.style.display = 'none';

  const body = {
    crop: document.getElementById('crop').value,
    location: document.getElementById('location').value,
    problem_description: document.getElementById('problem').value
  };

  try {
    const res = await fetch('/agent/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    resultEl.style.display = 'block';

    if (!data.ok) {
      resultEl.className = 'escalate';
      resultEl.textContent = data.message;
    } else {
      resultEl.className = data.escalated ? 'escalate' : '';
      const tag = data.escalated
        ? '<span class="label-tag tag-escalate">ESCALATE TO OFFICER</span>'
        : '<span class="label-tag tag-ok">Advice</span>';
      resultEl.innerHTML = tag + '<br><br>' +
        '<b>Crop advice:</b> ' + data.crop_advice + '<br><br>' +
        '<b>Market advice:</b> ' + data.market_advice + '<br><br>' +
        '<b>Recommended action:</b> ' + data.recommended_action;
    }
  } catch (e) {
    resultEl.style.display = 'block';
    resultEl.textContent = 'Something went wrong. Please try again.';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Get Advice';
  }
}
</script>
</body>
</html>
"""
