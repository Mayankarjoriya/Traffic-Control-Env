"""
Inference Script — AdaptiFlow Traffic Control
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import textwrap
from typing import List, Optional
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Import the OpenEnv client and action model (self-contained, no env.py needed)
from client import SmartTrafficEnv
from models import SmartTrafficAction


load_dotenv()

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")  or "Qwen/Qwen2.5-72B-Instruct"

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK    = "traffic_env"
MAX_STEPS    = 20
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.5

ACTIONS = ["NORTH_GREEN", "SOUTH_GREEN", "EAST_GREEN", "WEST_GREEN"]

TASK_NAMES = {
    1: "task_1_easy",
    2: "task_2_medium",
    3: "task_3_hard",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert traffic signal controller AI for a four-way intersection.
    Your goal is to minimize total waiting time across all lanes.

    IMPORTANT PRIORITIES (in order):
    1. AMBULANCE: If an ambulance is present, you MUST clear its lane immediately.
       Pick the action that includes the ambulance's lane. Every step the ambulance
       waits costs a -5.0 penalty.
    2. RUSH HOUR: The rush-hour lane gains 1-5 new cars every step. Prioritise
       clearing it frequently to prevent queue buildup.
    3. HIGH WAIT / HIGH CARS: Among remaining lanes, favour the pair with the
       highest combined car count and cumulative wait time.

    Green lights clear up to 8 cars per step from each green lane.
    Red lanes accumulate +10 wait-time units per step.

    ACTIONS:
      NORTH_GREEN — North gets green
      SOUTH_GREEN — South gets green
      EAST_GREEN  — East gets green
      WEST_GREEN  — West gets green

    THINK step-by-step:
    1. Check for ambulance — which lane? Which action clears it?
    2. Check rush-hour lane — does it need immediate attention?
    3. Compare car counts and wait times across all four lanes.
    4. Pick the single best action.

    After your reasoning, output your final decision wrapped in XML tags:
    <action>ACTION_NAME</action>
""").strip()


# ── Logging functions (mandatory format) ─────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_prompt(step: int, state: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        ┌──────────┬───────┬────────────┐
        │  Lane    │ Cars  │ Wait Time  │
        ├──────────┼───────┼────────────┤
        │  North   │ {state['north_cars']:>5} │ {state['north_wait']:>10} │
        │  South   │ {state['south_cars']:>5} │ {state['south_wait']:>10} │
        │  East    │ {state['east_cars']:>5} │ {state['east_wait']:>10} │
        │  West    │ {state['west_cars']:>5} │ {state['west_wait']:>10} │
        └──────────┴───────┴────────────┘
        Ambulance present : {state['ambulance']}  (lane: {state['ambulance_lane']})
        Rush-hour lane    : {state['rush_hour']}

        Recent history:
        {history_block}

        Think step-by-step, then give your final answer as <action>ACTION_NAME</action>.
    """).strip()


# ── Model action function ─────────────────────────────────────────────────────

def _extract_action(raw: str) -> str:
    """Extract the action from model output using layered fallbacks."""
    # 1. Try <action>...</action> XML tags
    match = re.search(r"<action>\s*(\w+)\s*</action>", raw, re.IGNORECASE)
    if match and match.group(1).upper() in ACTIONS:
        return match.group(1).upper()

    # 2. Scan for any valid action keyword in the raw output
    for action in ACTIONS:
        if action in raw.upper():
            return action

    # 3. Last resort
    return "NORTH_GREEN"


def get_model_action(client: OpenAI, step: int, state: dict, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, state, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = completion.choices[0].message.content or ""
        return _extract_action(raw)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "NORTH_GREEN"  # fallback


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(openai_client: OpenAI, task_id: int) -> None:
    task_name = TASK_NAMES[task_id]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    history: List[str]   = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    with SmartTrafficEnv(base_url=ENV_URL).sync() as env:
        result = env.reset(task_id=task_id)
        state = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            action_str = get_model_action(openai_client, step, state, history)

            action = SmartTrafficAction(action=action_str, task_id=task_id)
            result = env.step(action)

            state = result.observation.model_dump()
            reward = result.reward
            done = result.done

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        # Scoring (using final state)
        overall_avg = (
            state["north_wait"] + state["south_wait"] +
            state["east_wait"]  + state["west_wait"]
        ) / 4
        score = min(max(1.0 - (overall_avg / 180), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in [1, 2, 3]:
        run_task(client, task_id)
