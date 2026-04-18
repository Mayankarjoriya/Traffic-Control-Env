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
ACTIONS = ["NORTH_GREEN", "SOUTH_GREEN", "EAST_GREEN", "WEST_GREEN"]

# Per-task acceptable wait-time thresholds.
# These represent the average wait time at which score becomes 0.
# Calibrated for gradual clearing (GREEN_CAPACITY = 8).
# Higher values for harder tasks that naturally take longer episodes.
ACCEPTABLE_WAIT_TIME = {
    1: 80,      # Easy: small queues (1-10), should finish quickly, low wait acceptable
    2: 240,     # Medium: medium queues (10-20), longer episode, more wait expected
    3: 220,     # Hard: large queues (15-30), but ambulance rule ends early, mid-range wait
}


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


def calculate_score(history: List[dict], task_id: int) -> float:
    """
    Average wait-time based score, clamped to [0, 1].
    
    Task-aware normalization: each task has its own acceptable wait-time threshold.
    Score = 1.0 when average wait is 0.
    Score = 0.0 when average wait reaches the task's threshold.
    
    Args:
        history: List of observation dicts from one episode.
        task_id: The task difficulty level (1, 2, or 3).
    
    Returns:
        Score in range [0.0, 1.0].
    """
    # Calculate average wait time per lane
    north_avg = sum(s["north_wait"] for s in history) / len(history)
    south_avg = sum(s["south_wait"] for s in history) / len(history)
    east_avg  = sum(s["east_wait"] for s in history) / len(history)
    west_avg  = sum(s["west_wait"] for s in history) / len(history)

    # Overall average across all four lanes
    overall_avg = (north_avg + south_avg + east_avg + west_avg) / 4
    
    # Get the acceptable wait time for this task (default to 150 if task_id unknown)
    threshold = ACCEPTABLE_WAIT_TIME.get(task_id, 150)
    
    # Normalize: score = 1.0 - (actual / threshold), clamped to [0, 1]
    normalized_score = overall_avg / threshold
    return max(0.0, min(1.0, 1.0 - normalized_score))


def grade_all(action_fn: Callable[[dict], str]) -> None:
    """Grade all 3 tasks and print scores."""
    for task_id in [1, 2, 3]:
        history = collect_history(task_id, action_fn)
        # Pass task_id to calculate_score so it uses the right threshold
        score = calculate_score(history, task_id)
        print(f"task_{task_id}: {score:.2f}")


if __name__ == "__main__":
    # Simple round-robin baseline that cycles through all four phases
    _cycle = itertools.cycle(ACTIONS)

    def baseline_action(state: dict) -> str:
        return next(_cycle)

    grade_all(baseline_action)