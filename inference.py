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

# --- Naya Dynamic URL Logic ---
parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, default=os.getenv("OPENENV_URL", "http://localhost:8000"))
args, _ = parser.parse_known_args()
ENV_URL = args.url
# ------------------------------

BENCHMARK    = "traffic_env"
MAX_STEPS    = 20
TEMPERATURE  = 0.0
MAX_TOKENS   = 10
SUCCESS_SCORE_THRESHOLD = 0.5

ACTIONS = ["NS_GREEN", "EW_GREEN", "NE_GREEN", "NW_GREEN"]

TASK_NAMES = {
    1: "task_1_easy",
    2: "task_2_medium",
    3: "task_3_hard",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a traffic signal controller AI.
    You will see the current traffic state with car counts and wait times for each lane.
    Your goal is to minimize total waiting time across all lanes.
    If an ambulance is present, prioritize clearing its lane immediately.
    Choose exactly one action from: NS_GREEN, EW_GREEN, NE_GREEN, NW_GREEN
    Reply with ONLY the action name, nothing else.
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
        North lane : {state['north_cars']} cars, waited {state['north_wait']} seconds
        South lane : {state['south_cars']} cars, waited {state['south_wait']} seconds
        East  lane : {state['east_cars']}  cars, waited {state['east_wait']}  seconds
        West  lane : {state['west_cars']}  cars, waited {state['west_wait']}  seconds
        Ambulance  : {state['ambulance']} (lane: {state['ambulance_lane']})
        Rush hour  : {state['rush_hour']}

        Previous steps:
        {history_block}

        Choose one action: NS_GREEN, EW_GREEN, NE_GREEN, NW_GREEN
        Reply with ONLY the action name.
    """).strip()


# ── Model action function ─────────────────────────────────────────────────────

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
        action = (completion.choices[0].message.content or "").strip()
        return action if action in ACTIONS else "NS_GREEN"  # fallback
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "NS_GREEN"  # fallback


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
