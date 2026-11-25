import os, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set the GROQ_API_KEY environment variable first.")
client = Groq(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # add your real site URL here after you deploy the front end
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are an expert Prompt Engineering evaluator.

Evaluate the user's prompt using these six criteria:

1. Role
2. Goal
3. Context
4. Constraints
5. Format
6. Style

Give each criterion a score from 0 to 10.

Provide:
- score
- explanation
- strengths
- weaknesses
- recommendations
- optimized_prompt

Return ONLY valid JSON.

JSON structure:

{
    "scores": {"Role": 0, "Goal": 0, "Context": 0, "Constraints": 0, "Format": 0, "Style": 0},
    "explanations": {"Role": "", "Goal": "", "Context": "", "Constraints": "", "Format": "", "Style": ""},
    "strengths": [],
    "weaknesses": [],
    "recommendations": [],
    "optimized_prompt": ""
}
"""

class EvalRequest(BaseModel):
    prompt: str
    scenario: str
    difficulty: str

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    if len(req.prompt) > 4000:
        raise HTTPException(413, "Prompt too long")
    user_prompt = f"""
Scenario: {req.scenario}

Difficulty: {req.difficulty}

Evaluate this prompt:

{req.prompt}
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        raise HTTPException(502, "Evaluation failed, please try again")
