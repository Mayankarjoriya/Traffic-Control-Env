# Smart Traffic Environment

An [OpenEnv](https://openenv.ai/)-compatible AI environment for traffic signal control optimization at a simulated four-way intersection.

---

## Environment Description & Motivation

Urban traffic signals are typically fixed-cycle, making them blind to real-time conditions. An AI agent with full intersection visibility can dramatically reduce average wait times by dynamically choosing which lanes to service on each cycle.

`smart_traffic_env` provides the training and evaluation ground for such an agent. At every timestep, the agent sees the number of cars queued in each of four lanes together with their cumulative wait times, and selects a signal phase that clears two chosen lanes while holding the other two at red. Special events — rush-hour surges and emergency ambulances — add realism and require the agent to balance competing priorities.

---

## Action Space

The agent picks **one** signal phase per step.

| Action | Lanes Receiving Green |
|--------|-----------------------|
| `NS_GREEN` | North + South |
| `EW_GREEN` | East + West |
| `NE_GREEN` | North + East |
| `NW_GREEN` | North + West |

Green lanes have their car count instantly cleared to zero. Red lanes accumulate `+10` wait-time units and retain their car counts (plus any new arrivals from rush-hour).

**Action payload schema:**
```json
{
  "action": "NS_GREEN",
  "task_id": 1
}
```

---

## Observation Space

Returned by the server after every `reset` and `step`:

| Field | Type | Description |
|-------|------|-------------|
| `north_cars` | `int ≥ 0` | Cars queued in the north lane |
| `south_cars` | `int ≥ 0` | Cars queued in the south lane |
| `east_cars` | `int ≥ 0` | Cars queued in the east lane |
| `west_cars` | `int ≥ 0` | Cars queued in the west lane |
| `north_wait` | `int ≥ 0` | Cumulative wait time for north lane (time units) |
| `south_wait` | `int ≥ 0` | Cumulative wait time for south lane (time units) |
| `east_wait` | `int ≥ 0` | Cumulative wait time for east lane (time units) |
| `west_wait` | `int ≥ 0` | Cumulative wait time for west lane (time units) |
| `current_green_0` | `str \| null` | First lane on green this step |
| `current_green_1` | `str \| null` | Second lane on green this step |
| `ambulance` | `bool` | Whether an ambulance is currently blocked |
| `ambulance_lane` | `str \| null` | Lane the ambulance is waiting in |
| `rush_hour` | `str \| null` | Lane receiving continuous new arrivals |
| `task_id` | `int` | Current difficulty level (1, 2, or 3) |
| `reward` | `float` | Reward earned this step |
| `done` | `bool` | Episode termination flag |

---

## Task Descriptions

### Task 1 — Easy ⭐

| Parameter | Value |
|-----------|-------|
| Initial cars / lane | 1–10 (random) |
| Rush hour | ❌ None |
| Ambulance | ❌ None |
| Done condition | All lanes = 0 cars |

A straightforward clearing task. The intersection starts lightly loaded and there are no dynamic arrivals. A greedy (always clear the busiest lane) or round-robin policy reliably clears it in under 20 steps.

---

### Task 2 — Medium ⭐⭐

| Parameter | Value |
|-----------|-------|
| Initial cars / lane | 10–20 (random) |
| Rush hour | ✅ One random lane, +1–5 cars/step |
| Ambulance | ❌ None |
| Done condition | All lanes < 5 cars |

The rush-hour lane never fully empties — it refills continuously. An agent must learn to revisit it frequently enough to prevent it from dominating the intersection while still servicing the other lanes.

---

### Task 3 — Hard ⭐⭐⭐

| Parameter | Value |
|-----------|-------|
| Initial cars / lane | 15–30 (random) |
| Rush hour | ✅ One random lane, +1–5 cars/step |
| Ambulance | ✅ Blocked in one random lane (−5.0 reward/step until cleared) |
| Done condition | All lanes < 15 cars **AND** ambulance cleared |

The heaviest scenario. The ambulance creates urgency — delaying it is expensive. The agent must prioritize the ambulance lane quickly while keeping the rest of the intersection from seizing up under rush-hour pressure.

---

## Reward Function

```
step_reward = Σ (−0.1 × cars_in_each_red_lane)
            − 5.0   [if ambulance is present and NOT cleared this step]
```

**Episode score** (used by the grader):
```
score = clamp(1.0 − (mean_wait / 180), 0.0, 1.0)
```

where `mean_wait` is the average of the four lane wait-times at episode end. A score ≥ 0.50 is considered a successful episode.

---

## Baseline Scores

Measured using a **round-robin baseline** (cycling `NS_GREEN → EW_GREEN → NE_GREEN → NW_GREEN`, max 100 steps):

| Task | Baseline Policy | Approx. Score |
|------|----------------|---------------|
| Task 1 — Easy | Round-robin | ~0.72 |
| Task 2 — Medium | Round-robin | ~0.45 |
| Task 3 — Hard | Round-robin | ~0.18 |

> Scores are stochastic due to random initial states. These values are approximate averages over multiple runs. A strong agent is expected to score ≥ 0.80 on Task 1, ≥ 0.60 on Task 2, and ≥ 0.40 on Task 3.

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) — or Docker

---

### Install Dependencies

```bash
cd smart_traffic_env
uv sync
```

---

### Start the Environment Server

**Local (uv):**
```bash
uv run --project . server
# or
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

**Docker:**
```bash
sudo docker build -t smart_traffic_env .
sudo docker run -p 8000:8000 smart_traffic_env
```

The server starts at `http://localhost:8000`.

---

### Configure LLM Credentials

Copy `.env.example` or edit `.env` directly:

```bash
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_URL=http://localhost:8000
```

---

### Run the LLM Inference Agent

```bash
uv run --project . python inference.py
```

The script runs all three tasks in order and prints per-step logs:
```
[START] task=task_1_easy env=traffic_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=EW_GREEN reward=-1.50 done=false error=null
...
[END]   success=true steps=8 score=0.843 rewards=-1.50,-0.80,...
```

---

### Run the Baseline Grader

```bash
uv run --project . python graders.py
```

Output:
```
task_1: 0.72
task_2: 0.45
task_3: 0.18
```

---

### Validate Submission

```bash
bash validate-submission.sh
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment. Body: `{"task_id": 1}` |
| `POST` | `/step` | Take one step. Body: `SmartTrafficAction` JSON |
| `GET` | `/state` | Return current state snapshot |
| `GET` | `/schema` | Return action & observation JSON schemas |
| `GET` | `/health` | Health check (`200 OK`) |
| `WS` | `/ws` | WebSocket endpoint for persistent sessions |