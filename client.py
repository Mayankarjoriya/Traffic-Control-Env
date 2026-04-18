# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Traffic Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SmartTrafficAction, SmartTrafficObservation
except ImportError:
    from models import SmartTrafficAction, SmartTrafficObservation


class SmartTrafficEnv(
    EnvClient[SmartTrafficAction, SmartTrafficObservation, State]
):
    """
    Client for the Smart Traffic Control Environment.

    Example:
        >>> with SmartTrafficEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(task_id=1)
        ...     print(result.observation)
        ...
        ...     action = SmartTrafficAction(action="NORTH_GREEN", task_id=1)
        ...     result = client.step(action)
        ...     print(result.observation.north_cars)
    """

    def _step_payload(self, action: SmartTrafficAction) -> Dict:
        """
        Convert SmartTrafficAction to JSON payload for step message.

        Args:
            action: SmartTrafficAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[SmartTrafficObservation]:
        """
        Parse server response into StepResult[SmartTrafficObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SmartTrafficObservation
        """
        obs_data = payload.get("observation", {})
        observation = SmartTrafficObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
