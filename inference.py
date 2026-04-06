import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from env import TrafficEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
BENCHMARK = "traffic_env"
MAX_STEPS = 100
SUCCESS_SCORE_THRESHOLD = 0.5

ACTIONS = ["NS_GREEN", "EW_GREEN", "NE_GREEN", "NW_GREEN"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a traffic signal controller AI.
    You will see the current traffic state with car counts and wait times for each lane.
    Your goal is to minimize total waiting time across all lanes.
    Choose exactly one action from: NS_GREEN, EW_GREEN, NE_GREEN, NW_GREEN
    Reply with ONLY the action name, nothing else.
""").strip()


# ── Logging functions (mandatory format) ─────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(state: dict) -> str:
    return textwrap.dedent(f"""
        Current traffic state:
        North lane: {state['north_cars']} cars, waited {state['north_wait']} seconds
        South lane: {state['south_cars']} cars, waited {state['south_wait']} seconds
        East lane:  {state['east_cars']} cars, waited {state['east_wait']} seconds
        West lane:  {state['west_cars']} cars, waited {state['west_wait']} seconds
        Ambulance present: {state['ambulance']} (lane: {state['ambulance_lane']})
        
        Choose one action from: NS_GREEN, EW_GREEN, NE_GREEN, NW_GREEN
        Reply with ONLY the action name.
    """).strip()


# ── GPT action function ───────────────────────────────────────────────────────

def get_action(client: OpenAI, state: dict) -> str:
    prompt = build_prompt(state)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # deterministic
            max_tokens=10,
        )
        action = completion.choices[0].message.content.strip()
        if action not in ACTIONS:
            return "NS_GREEN"  # fallback
        return action
    except Exception as e:
        print(f"[DEBUG] API error: {e}", flush=True)
        return "NS_GREEN"  # fallback


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: int):
    task_name = f"task_{task_id}"
    env = TrafficEnv()
    state = env.reset(task_id)

    rewards: List[float] = []
    history = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, state)
            state, reward, done, info = env.step(action)

            rewards.append(reward)
            steps_taken = step
            history.append(state)

            log_step(step=step, action=action, reward=reward, done=done, error=None)

            if done:
                break

        # Score calculate karo grader logic se
        if history:
            north_avg = sum(s["north_wait"] for s in history) / len(history)
            south_avg = sum(s["south_wait"] for s in history) / len(history)
            east_avg  = sum(s["east_wait"]  for s in history) / len(history)
            west_avg  = sum(s["west_wait"]  for s in history) / len(history)
            overall_avg = (north_avg + south_avg + east_avg + west_avg) / 4
            score = 1.0 - (overall_avg - 10) / 170
            score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id in [1, 2, 3]:
        run_task(client, task_id)