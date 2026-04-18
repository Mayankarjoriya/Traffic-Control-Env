# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Traffic Control Environment Implementation (Enhanced).

Ported from env.py (TrafficEnv).  The agent selects a traffic-signal phase
each step; the environment updates car counts, wait times, and returns a
reward signal together with a done flag.

Difficulty levels (task_id):
    1 – Easy   : small queues, no rush hour, no ambulance
    2 – Medium : bigger queues, multiple rush hour lanes
    3 – Hard   : large queues, multiple rush hour lanes, ambulance in random lane

ENHANCEMENTS:
    - Red light violations: Impatient vehicles cross red signals if wait exceeds threshold
    - Multi-lane rush hour: Task 2 and 3 now have 2-3 lanes in rush simultaneously
    - Realistic congestion: Agent must balance fairness with efficiency
"""
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartTrafficAction, SmartTrafficObservation
except ImportError:
    from models import SmartTrafficAction, SmartTrafficObservation


class SmartTrafficEnvironment(Environment):
    """
    Four-way intersection traffic-control environment with violations and multi-lane rush.

    The agent picks one of four signal phases per step.  The chosen pair of
    lanes clears up to GREEN_CAPACITY cars each while the opposing lanes
    accumulate wait time.  Multiple rush-hour lanes (task ≥ 2) add random new
    arrivals every step.  An ambulance (task 3) penalises phases that leave
    its lane red.  Vehicles that wait too long violate red signals, creating
    additional penalty pressure.

    Episode ends when all lane queues fall below the task-specific threshold.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Maximum cars cleared per green lane per step (realistic flow model)
    GREEN_CAPACITY: int = 8

    # Red light violation thresholds
    RED_LIGHT_VIOLATION_THRESHOLD: int = 70  # If wait > 50, vehicles start violating
    RED_LIGHT_VIOLATION_RATE: float = 0.015   # 3% of waiting cars violate per step

    # Map: action string → (green_0, green_1, lanes_that_clear, lanes_that_wait)
    _PHASE_MAP = {
        "NORTH_GREEN": ("north", None, ["north"], ["south", "east", "west"]),
        "SOUTH_GREEN": ("south", None, ["south"], ["north", "east", "west"]),
        "EAST_GREEN":  ("east", None,  ["east"],  ["north", "south", "west"]),
        "WEST_GREEN":  ("west", None,  ["west"],  ["north", "south", "east"]),
    }

    def __init__(self):
        """Initialise with a blank internal state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env_state: dict = {}   # mirrors the old self.state dict

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 1) -> SmartTrafficObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Difficulty level (1 = Easy, 2 = Medium, 3 = Hard).

        Returns:
            SmartTrafficObservation with the initial intersection state.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ---- Easy -------------------------------------------------------
        if task_id == 1:
            self._env_state = {
                "north_cars":      random.randint(1, 10),
                "south_cars":      random.randint(1, 10),
                "east_cars":       random.randint(1, 10),
                "west_cars":       random.randint(1, 10),
                "north_wait":      0,
                "south_wait":      0,
                "east_wait":       0,
                "west_wait":       0,
                "current_green_0": None,
                "current_green_1": None,
                "ambulance":       False,
                "ambulance_lane":  None,
                "rush_hour":       [],  # No rush hour in easy
                "task_id":         1,
            }

        # ---- Medium (ENHANCED: Multiple rush lanes) --------------------
        elif task_id == 2:
            self._env_state = {
                "north_cars":      random.randint(10, 20),
                "south_cars":      random.randint(10, 20),
                "east_cars":       random.randint(10, 20),
                "west_cars":       random.randint(10, 20),
                "north_wait":      0,
                "south_wait":      0,
                "east_wait":       0,
                "west_wait":       0,
                "current_green_0": None,
                "current_green_1": None,
                "ambulance":       False,
                "ambulance_lane":  None,
                # ENHANCED: 2 random lanes in rush hour (not just 1)
                "rush_hour":       random.sample(["north", "south", "east", "west"], k=2),
                "task_id":         2,
            }

        # ---- Hard (ENHANCED: Multiple rush lanes + ambulance) ---------
        elif task_id == 3:
            self._env_state = {
                "north_cars":      random.randint(15, 30),
                "south_cars":      random.randint(15, 30),
                "east_cars":       random.randint(15, 30),
                "west_cars":       random.randint(15, 30),
                "north_wait":      0,
                "south_wait":      0,
                "east_wait":       0,
                "west_wait":       0,
                "current_green_0": None,
                "current_green_1": None,
                "ambulance":       True,
                "ambulance_lane":  random.choice(["north", "south", "east", "west"]),
                # ENHANCED: 2-3 lanes in heavy rush hour (more realistic peak traffic)
                "rush_hour":       random.sample(
                    ["north", "south", "east", "west"], 
                    k=random.randint(2, 3)
                ),
                "task_id":         3,
            }

        else:
            raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")

        return self._build_observation(reward=0.0, done=False)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: SmartTrafficAction) -> SmartTrafficObservation:  # type: ignore[override]
        """
        Execute one traffic-signal phase with violations and multi-lane rush support.

        Args:
            action: SmartTrafficAction specifying the phase and task_id.

        Returns:
            SmartTrafficObservation containing updated state, reward, and done flag.
        """
        # --- BULLETPROOF TASK SYNC PATCH ---
        if self._state.step_count == 0 and self._env_state.get("task_id") != action.task_id:
            self.reset(task_id=action.task_id)

        self._state.step_count += 1

        phase = action.action

        # If the validator sends random test data, safely default to NORTH_GREEN
        if phase not in self._PHASE_MAP:
            phase = "NORTH_GREEN"
            
        task_id = self._env_state.get("task_id", 1)

        green_0, green_1, clear_lanes, wait_lanes = self._PHASE_MAP[phase]

        # ---- Step 1 : update signal state --------------------------------
        self._env_state["current_green_0"] = green_0
        self._env_state["current_green_1"] = green_1

        # ---- Step 2 : compute reward (cars still blocked cost 0.05 each) --
        # Reduced from 0.1 to 0.05 because 3 lanes wait now (single-lane green)
        reward = 0.0
        for lane in wait_lanes:
            reward -= self._env_state[f"{lane}_cars"] * 0.05

        # ---- Step 3 : accumulate wait time for red lanes -----------------
        for lane in wait_lanes:
            self._env_state[f"{lane}_wait"] += 10
            
        # Small penalty for total accumulated wait (gentle pressure to keep wait low)
        total_wait_now = sum(self._env_state[f"{l}_wait"] for l in ["north", "south", "east", "west"])
        reward -= 0.01 * total_wait_now

        # ---- Step 4 : graduated clearing of green lanes ------------------
        for lane in clear_lanes:
            current = self._env_state[f"{lane}_cars"]
            self._env_state[f"{lane}_cars"] = max(0, current - self.GREEN_CAPACITY)

        # ---- Step 5 : rush-hour arrivals (multiple lanes) ----------------
        rush_lanes = self._env_state.get("rush_hour", [])
        if rush_lanes:  # Now it's a list that can have multiple lanes
            for rush_lane in rush_lanes:
                self._env_state[f"{rush_lane}_cars"] += random.randint(1, 5)

        # ---- Step 6 : ambulance penalty and clearing ---------------------
        if self._env_state.get("ambulance"):
            if self._env_state.get("ambulance_lane") not in [green_0, green_1]:
                reward -= 5.0  # ambulance is still waiting
            else:
                # Ambulance got green light, so it clears the intersection
                self._env_state["ambulance"] = False
                self._env_state["ambulance_lane"] = None

        # ---- Step 7 : red light violations (impatient vehicles crossing) -------
        violation_penalty = 0.0
        for lane in ["north", "south", "east", "west"]:
            violations = self._calculate_violations(lane)
            if violations > 0:
                # Vehicles crossed red light—creates safety hazard and disruption
                violation_penalty -= violations * 2.0  # Heavy penalty per violation
                # Remove violating vehicles (they escaped but caused disruption)
                self._env_state[f"{lane}_cars"] -= violations
                # Reset their wait time (they're gone)
                self._env_state[f"{lane}_wait"] = 0

        reward += violation_penalty

        # ---- Step 8 : done check (per-task threshold) --------------------
        n = self._env_state["north_cars"]
        s = self._env_state["south_cars"]
        e = self._env_state["east_cars"]
        w = self._env_state["west_cars"]

        if task_id == 1:
            done = (n == 0 and s == 0 and e == 0 and w == 0)
        elif task_id == 2:
            done = (n < 5 and s < 5 and e < 5 and w < 5)
        elif task_id == 3:
            # Episode finishes when cars are minimal AND ambulance has successfully left
            ambulance_cleared = not self._env_state.get("ambulance", False)
            done = (n < 15 and s < 15 and e < 15 and w < 15) and ambulance_cleared
        else:
            print(f"No Task Assigned: {task_id}")
            done = False

        total_wait = (
            self._env_state["north_wait"]
            + self._env_state["south_wait"]
            + self._env_state["east_wait"]
            + self._env_state["west_wait"]
        )

        return self._build_observation(
            reward=reward,
            done=done,
            metadata={"total_wait": total_wait, "step": self._state.step_count},
        )

    # ------------------------------------------------------------------
    # state property  (required by Environment interface)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the internal openenv State object."""
        return self._state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _calculate_violations(self, lane: str) -> int:
        """
        Calculate how many vehicles violate the red light due to impatience.
        
        If a lane has been waiting too long (exceeds RED_LIGHT_VIOLATION_THRESHOLD),
        some percentage of vehicles will cross the red signal. This creates pressure
        on the agent to rotate signals fairly and prevent excessive wait times.
        
        Args:
            lane: The lane name ("north", "south", "east", "west")
        
        Returns:
            Number of vehicles crossing the red light
        """
        wait_time = self._env_state[f"{lane}_wait"]
        
        # Only violations if wait exceeds threshold
        if wait_time <= self.RED_LIGHT_VIOLATION_THRESHOLD:
            return 0
        
        # Calculate violation count: more wait = more violations
        excess_wait = wait_time - self.RED_LIGHT_VIOLATION_THRESHOLD
        violation_count = int(excess_wait * self.RED_LIGHT_VIOLATION_RATE)
        
        # Can't violate more cars than exist in the queue
        current_cars = self._env_state[f"{lane}_cars"]
        return min(violation_count, current_cars)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> SmartTrafficObservation:
        """
        Construct a SmartTrafficObservation from the current env state.
        
        Converts internal rush_hour list to comma-separated string for observation.
        """
        s = self._env_state
        
        # Handle rush_hour: it's a list, convert to comma-separated string
        rush_hour_str = None
        if s.get("rush_hour"):
            rush_hour_str = ",".join(s.get("rush_hour", []))
        
        return SmartTrafficObservation(
            # car counts
            north_cars=s.get("north_cars", 0),
            south_cars=s.get("south_cars", 0),
            east_cars=s.get("east_cars",  0),
            west_cars=s.get("west_cars",  0),
            # wait times
            north_wait=s.get("north_wait", 0),
            south_wait=s.get("south_wait", 0),
            east_wait=s.get("east_wait",  0),
            west_wait=s.get("west_wait",  0),
            # signal
            current_green_0=s.get("current_green_0"),
            current_green_1=s.get("current_green_1"),
            # special conditions
            ambulance=s.get("ambulance", False),
            ambulance_lane=s.get("ambulance_lane"),
            rush_hour=rush_hour_str,  # Now a comma-separated string or None
            # episode meta
            task_id=s.get("task_id", 1),
            # openenv base fields
            reward=reward,
            done=done,
            metadata=metadata or {},
        )