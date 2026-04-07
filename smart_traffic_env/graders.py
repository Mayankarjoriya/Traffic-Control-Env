"""
Grader Script — Smart Traffic Control Environment.

Uses the SmartTrafficEnv client to connect to the running server,
run episodes with a baseline action policy, and compute scores.
No dependency on env.py — fully self-contained.
"""

import itertools
from typing import Callable, Dict, List

from client import SmartTrafficEnv
from models import SmartTrafficAction


ENV_URL = "http://localhost:8000"
ACTIONS = ["NS_GREEN", "EW_GREEN", "NE_GREEN", "NW_GREEN"]


def collect_history(
    task_id: int,
    action_fn: Callable[[dict], str],
    max_steps: int = 100,
) -> List[dict]:
    """Run one episode via the server client, return list of observation dicts."""
    history: List[dict] = []

    with SmartTrafficEnv(base_url=ENV_URL).sync() as env:
        result = env.reset(task_id=task_id)
        state = result.observation.model_dump()

        for _ in range(max_steps):
            action_str = action_fn(state)
            action = SmartTrafficAction(action=action_str, task_id=task_id)
            result = env.step(action)

            state = result.observation.model_dump()
            history.append(state)

            if result.done:
                break

    return history


def calculate_score(history: List[dict]) -> float:
    """Average wait-time based score, clamped to [0, 1]."""
    north_avg = sum(s["north_wait"] for s in history) / len(history)
    south_avg = sum(s["south_wait"] for s in history) / len(history)
    east_avg  = sum(s["east_wait"] for s in history) / len(history)
    west_avg  = sum(s["west_wait"] for s in history) / len(history)

    overall_avg = (north_avg + south_avg + east_avg + west_avg) / 4
    return max(0.0, min(1.0, 1.0 - (overall_avg / 180)))


def grade_all(action_fn: Callable[[dict], str]) -> None:
    """Grade all 3 tasks and print scores."""
    for task_id in [1, 2, 3]:
        history = collect_history(task_id, action_fn)
        score = calculate_score(history)
        print(f"task_{task_id}: {score:.2f}")


if __name__ == "__main__":
    # Simple round-robin baseline that cycles through all four phases
    _cycle = itertools.cycle(ACTIONS)

    def baseline_action(state: dict) -> str:
        return next(_cycle)

    grade_all(baseline_action)
