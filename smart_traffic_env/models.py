# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Traffic Env Environment.

Actions map to traffic signal phases (which two lanes get the green light).
Observations expose the full intersection state: car counts, wait times,
ambulance presence, rush-hour lane, and the current green lanes.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SmartTrafficAction(Action):
    """
    Choose which pair of opposing lanes should receive the green light.

    Allowed values mirror the ACTIONS list from the original env.py:
        - ``NS_GREEN`` : North + South green
        - ``EW_GREEN`` : East + West green
        - ``NE_GREEN`` : North + East green
        - ``NW_GREEN`` : North + West green
    """

    action: Literal["NS_GREEN", "EW_GREEN", "NE_GREEN", "NW_GREEN"] = Field(
        ...,
        description=(
            "Traffic-signal phase to activate. "
            "One of: NS_GREEN, EW_GREEN, NE_GREEN, NW_GREEN."
        ),
    )

    task_id: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Difficulty level: 1 = Easy, 2 = Medium, 3 = Hard.",
    )


# ---------------------------------------------------------------------------
# Observation  (also serves as the step response payload)
# ---------------------------------------------------------------------------

class SmartTrafficObservation(Observation):
    """
    Full intersection state returned after every reset / step call.

    Car counts reflect how many vehicles are *waiting* in each lane.
    Wait times accumulate across steps for lanes that are held at red.
    """

    # --- car counts ---
    north_cars: int = Field(default=0, ge=0, description="Cars waiting in the north lane.")
    south_cars: int = Field(default=0, ge=0, description="Cars waiting in the south lane.")
    east_cars:  int = Field(default=0, ge=0, description="Cars waiting in the east lane.")
    west_cars:  int = Field(default=0, ge=0, description="Cars waiting in the west lane.")

    # --- cumulative wait times (in seconds / time-steps) ---
    north_wait: int = Field(default=0, ge=0, description="Cumulative wait time for the north lane.")
    south_wait: int = Field(default=0, ge=0, description="Cumulative wait time for the south lane.")
    east_wait:  int = Field(default=0, ge=0, description="Cumulative wait time for the east lane.")
    west_wait:  int = Field(default=0, ge=0, description="Cumulative wait time for the west lane.")

    # --- signal state ---
    current_green_0: Optional[str] = Field(
        default=None,
        description="First lane currently on green (north / south / east / west).",
    )
    current_green_1: Optional[str] = Field(
        default=None,
        description="Second lane currently on green (north / south / east / west).",
    )

    # --- special conditions ---
    ambulance: bool = Field(
        default=False,
        description="Whether an ambulance is present at the intersection.",
    )
    ambulance_lane: Optional[str] = Field(
        default=None,
        description="Lane the ambulance is waiting in (if ambulance=True).",
    )
    rush_hour: Optional[str] = Field(
        default=None,
        description="Lane experiencing rush-hour traffic (cars added each step), or None.",
    )

    # --- episode metadata ---
    task_id: int = Field(
        default=1,
        description="Active difficulty level: 1 = Easy, 2 = Medium, 3 = Hard.",
    )
