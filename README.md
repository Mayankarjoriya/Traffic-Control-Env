# Traffic-Control-Env

A multi-task reinforcement learning environment for AI-driven traffic signal optimization at a four-way intersection, compatible with the [OpenEnv](https://openenv.ai/) framework.

---

## Overview

Urban traffic congestion is a major modern challenge, increasing commute times, fuel consumption, and emissions. Traditional fixed-cycle traffic lights are inefficient — they cannot adapt to real-time conditions such as uneven traffic load, rush-hour surges, or emergency vehicles.

This environment allows AI agents to learn a dynamic traffic signal control policy for a simulated four-way intersection. The agent observes the current number of vehicles and cumulative wait times per lane, and must choose a signal phase each step to minimize total waiting time across all lanes.

The environment is built on the **OpenEnv** framework, exposing a clean REST/WebSocket API so any agent (LLM-based, RL-based, or rule-based) can interact with it without requiring direct Python imports.

---

## Repository Structure

```
Traffic-Control-Env/
└── smart_traffic_env/          # OpenEnv-compatible environment package
    ├── server/
    │   ├── app.py              # FastAPI application entry point
    │   └── smart_traffic_env_environment.py  # Core simulation logic
    ├── models.py               # Pydantic Action & Observation schemas
    ├── client.py               # Python client for the HTTP server
    ├── inference.py            # LLM agent inference script
    ├── graders.py              # Baseline grading script
    ├── openenv.yaml            # OpenEnv spec manifest
    ├── Dockerfile              # Multi-stage Docker build
    └── pyproject.toml          # Python project metadata & dependencies
```

---

## Environment Description

The environment simulates a single **four-way intersection** with four lanes: North, South, East, and West. Each lane has a queue of waiting vehicles. Each timestep, the agent selects a signal phase (which pair of lanes receives a green light), which clears the chosen lanes and accumulates wait time for the opposing lanes. The episode ends when the lane queues fall below the task-specific threshold.

### Special Conditions

| Condition | Description |
|-----------|-------------|
| **Rush Hour** | A designated lane receives 1–5 new cars every step due to high demand. |
| **Ambulance** | An emergency vehicle is blocked in one lane. Each step the ambulance is not given a green light incurs a penalty of **−5.0** to the reward. Once cleared, it does not return. |

---

## Action Space

The agent submits one discrete signal phase per step.

| Action | Lanes Getting Green |
|--------|---------------------|
| `NS_GREEN` | North + South |
| `EW_GREEN` | East + West |
| `NE_GREEN` | North + East |
| `NW_GREEN` | North + West |

Each action payload also includes the `task_id` to indicate the active difficulty.

```json
{
  "action": "NS_GREEN",
  "task_id": 1
}
```

When a pair of lanes gets the green, those lanes have their car queues cleared to zero. The opposing lanes accumulate `+10` wait-time units and continue holding their vehicle counts.

---

## Observation Space

After every `reset` or `step` call, the server returns a full intersection state:

| Field | Type | Description |
|-------|------|-------------|
| `north_cars` | `int ≥ 0` | Cars waiting in the north lane |
| `south_cars` | `int ≥ 0` | Cars waiting in the south lane |
| `east_cars` | `int ≥ 0` | Cars waiting in the east lane |
| `west_cars` | `int ≥ 0` | Cars waiting in the west lane |
| `north_wait` | `int ≥ 0` | Cumulative wait time (units) for north lane |
| `south_wait` | `int ≥ 0` | Cumulative wait time (units) for south lane |
| `east_wait` | `int ≥ 0` | Cumulative wait time (units) for east lane |
| `west_wait` | `int ≥ 0` | Cumulative wait time (units) for west lane |
| `current_green_0` | `str \| null` | First lane currently on green |
| `current_green_1` | `str \| null` | Second lane currently on green |
| `ambulance` | `bool` | Whether an ambulance is present |
| `ambulance_lane` | `str \| null` | Lane the ambulance is waiting in |
| `rush_hour` | `str \| null` | Lane experiencing ongoing surge arrivals |
| `task_id` | `int` | Active difficulty level (1, 2, or 3) |
| `reward` | `float` | Reward received this step |
| `done` | `bool` | Whether the episode has terminated |

---

## Task Descriptions

There are three tasks of increasing difficulty:

### Task 1 — Easy
- **Initial cars per lane:** 1–10 (random)
- **Rush hour:** None
- **Ambulance:** None
- **Done condition:** All four lanes must reach exactly 0 cars
- **Expected difficulty:** An agent using a simple greedy strategy (clear the busiest lane) performs well. A random or round-robin policy can also succeed within a modest number of steps.

### Task 2 — Medium
- **Initial cars per lane:** 10–20 (random)
- **Rush hour:** One randomly selected lane receives 1–5 new cars each step
- **Ambulance:** None
- **Done condition:** All four lanes must drop below 5 cars
- **Expected difficulty:** The continuous arrivals in the rush-hour lane force the agent to revisit it regularly, rather than just clearing in sequence. A purely round-robin approach struggles.

### Task 3 — Hard
- **Initial cars per lane:** 15–30 (random)
- **Rush hour:** One randomly selected lane receives 1–5 new cars each step
- **Ambulance:** Present in a random lane; incurs **−5.0** reward per step until cleared
- **Done condition:** All four lanes below 15 cars AND the ambulance has been cleared
- **Expected difficulty:** The agent must balance two competing priorities: managing heavy traffic load while urgently clearing the ambulance lane. Naive policies that ignore the ambulance accumulate large negative rewards.

---

## Reward Function

Each step, the reward is computed as:

```
reward = -0.1 × (cars in waiting lanes)
       - 5.0   [if ambulance is present and not cleared this step]
```

The per-episode **score** is derived from the final average wait time across all four lanes:

```
score = clamp(1.0 - (mean_wait / 180), 0.0, 1.0)
```

A score of `1.0` means no waiting time accumulated; `0.0` means the mean wait reached or exceeded 180 time units.

---

## Baseline Scores

The following baseline scores were measured using a **round-robin policy** (cycling through `NS_GREEN → EW_GREEN → NE_GREEN → NW_GREEN` repeatedly, max 100 steps):

| Task | Policy | Score |
|------|--------|-------|
| Task 1 — Easy | Round-robin | ~0.72 |
| Task 2 — Medium | Round-robin | ~0.45 |
| Task 3 — Hard | Round-robin | ~0.18 |

> **Note:** Scores vary run-to-run due to randomized initial states. These are approximate averages.  
> A score ≥ 0.50 is considered a successful episode.

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager — or Docker

---

### Option A: Run with `uv` (local)

```bash
cd smart_traffic_env

# Install dependencies
uv sync

# Start the environment server (port 8000)
uv run --project . server
```

### Option B: Run with Docker

```bash
cd smart_traffic_env

# Build the Docker image
sudo docker build -t smart_traffic_env .

# Run the container
sudo docker run -p 8000:8000 smart_traffic_env
```

The server will be available at `http://localhost:8000`.

---

### Running the Inference Agent

Configure your LLM credentials in `smart_traffic_env/.env`:

```bash
HF_TOKEN=your_token_here
API_BASE_URL=https://router.huggingface.co/v1   # or your own endpoint
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_URL=http://localhost:8000
```

Then run the inference script:

```bash
cd smart_traffic_env
uv run --project . python inference.py
```

Output follows the mandatory format:
```
[START] task=task_1_easy env=traffic_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=NS_GREEN reward=-2.10 done=false error=null
...
[END]   success=true steps=12 score=0.821 rewards=-2.10,-1.80,...
```

---

### Running the Grader

```bash
cd smart_traffic_env
uv run --project . python graders.py
```

The grader runs all three tasks with the built-in round-robin baseline and prints a score per task.

---

### Validating Your Submission

```bash
cd smart_traffic_env
bash validate-submission.sh
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset the environment (`{"task_id": 1}`) |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Get current state without stepping |
| `GET` | `/schema` | Get action/observation JSON schemas |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket for persistent sessions |
