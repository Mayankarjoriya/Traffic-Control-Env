# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Traffic Control Environment Implementation.

Ported from env.py (TrafficEnv).  The agent selects a traffic-signal phase
each step; the environment updates car counts, wait times, and returns a
reward signal together with a done flag.

Difficulty levels (task_id):
    1 – Easy   : small queues, no rush hour, no ambulance
    2 – Medium : bigger queues, north rush hour
    3 – Hard   : large queues, north rush hour, ambulance in west lane
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
    Four-way intersection traffic-control environment.

    The agent picks one of four signal phases per step.  The chosen pair of
    lanes clears its queue (cars → 0) while the opposing lanes accumulate
    wait time.  A rush-hour lane (task ≥ 2) adds random new arrivals every
    step.  An ambulance (task 3) penalises phases that leave its lane red.

    Episode ends when all lane queues fall below the task-specific threshold.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Map: action string → (green_0, green_1, lanes_that_clear, lanes_that_wait)
    _PHASE_MAP = {
        "NS_GREEN": ("north", "south", ["north", "south"], ["east", "west"]),
        "EW_GREEN": ("east",  "west",  ["east",  "west"],  ["north", "south"]),
        "NE_GREEN": ("north", "east",  ["north", "east"],  ["south", "west"]),
        "NW_GREEN": ("north", "west",  ["north", "west"],  ["south", "east"]),
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

        Mirrors TrafficEnv.reset() from env.py.

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
                "rush_hour":       False,
                "task_id":         1,
            }

        # ---- Medium -----------------------------------------------------
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
                "rush_hour":       random.choice(["north", "south", "east", "west"]),
                "task_id":         2,
            }

        # ---- Hard -------------------------------------------------------
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
                "rush_hour":       random.choice(["north", "south", "east", "west"]),
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
        Execute one traffic-signal phase.

        Mirrors TrafficEnv.step() from env.py.

        Args:
            action: SmartTrafficAction specifying the phase and (optionally)
                    the task_id to switch difficulty mid-episode.

        Returns:
            SmartTrafficObservation containing updated state, reward, and
            done flag.
        """
        # --- BULLETPROOF TASK SYNC PATCH ---
        # If the HTTP client dropped the task_id during reset(), catch it 
        # on the very first step using the action payload and re-initialize.
        if self._state.step_count == 0 and self._env_state.get("task_id") != action.task_id:
            self.reset(task_id=action.task_id)


        self._state.step_count += 1

        phase     = action.action

        # If the validator sends random test data, safely default to NS_GREEN
        if phase not in self._PHASE_MAP:
            phase = "NS_GREEN"
            
        task_id   = self._env_state.get("task_id", 1)

        green_0, green_1, clear_lanes, wait_lanes = self._PHASE_MAP[phase]

        # ---- Step 1 : update signal state --------------------------------
        self._env_state["current_green_0"] = green_0
        self._env_state["current_green_1"] = green_1

        # ---- Step 2 : compute reward (cars still blocked cost 0.1 each) --
        reward = 0.0
        for lane in wait_lanes:
            reward -= self._env_state[f"{lane}_cars"] * 0.1

        # ---- Step 3 : accumulate wait time for red lanes -----------------
        for lane in wait_lanes:
            self._env_state[f"{lane}_wait"] += 10

        # ---- Step 4 : clear green lanes ----------------------------------
        for lane in clear_lanes:
            self._env_state[f"{lane}_cars"] = 0

        # ---- Step 5 : rush-hour arrivals ---------------------------------
        rush_lane = self._env_state.get("rush_hour")
        if rush_lane and rush_lane is not False:
            self._env_state[f"{rush_lane}_cars"] += random.randint(1, 5)

        # ---- Step 6 : ambulance penalty and clearing ---------------------
        if self._env_state.get("ambulance"):
            if self._env_state.get("ambulance_lane") not in [green_0, green_1]:
                reward -= 5.0  # ambulance is still waiting
            else:
                # Ambulance got green light, so it clears the intersection
                self._env_state["ambulance"] = False
                self._env_state["ambulance_lane"] = None

        # ---- Step 7 : done check (per-task threshold) --------------------
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

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> SmartTrafficObservation:
        """Construct a SmartTrafficObservation from the current env state."""
        s = self._env_state
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
            rush_hour=s.get("rush_hour") or None,
            # episode meta
            task_id=s.get("task_id", 1),
            # openenv base fields
            reward=reward,
            done=done,
            metadata=metadata or {},
        )
