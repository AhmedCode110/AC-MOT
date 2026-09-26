from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateDensityDecision:
    target_candidates: int
    raw_count_ema: float | None

    sensitivity: float
    low_threshold: float
    high_threshold: float
    new_track_threshold: float


class CandidateDensityController:
    """
    Universal causal overload protection.

    Goals:
    - no detector-specific constants
    - never make the base AC policy more permissive
    - prevent candidate explosions
    - current detector count affects future frames only
    """

    def __init__(
        self,
        *,
        raw_decay: float = 0.80,
        base_budget: float = 12.0,
        sci_budget: float = 36.0,
        min_budget: int = 12,
        max_budget: int = 48,
        min_sensitivity: float = 0.06,
    ):
        self.raw_decay = float(raw_decay)

        self.base_budget = float(base_budget)
        self.sci_budget = float(sci_budget)

        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)

        self.min_sensitivity = float(
            min_sensitivity
        )

        self.raw_count_ema = None

    def reset(self):
        self.raw_count_ema = None

    def _target_budget(
        self,
        sci: float,
    ) -> int:

        sci = float(
            np.clip(
                sci,
                0.0,
                1.0,
            )
        )

        target = int(
            round(
                self.base_budget
                +
                self.sci_budget * sci
            )
        )

        return max(
            self.min_budget,
            min(
                self.max_budget,
                target,
            ),
        )

    def decide(
        self,
        *,
        base_sensitivity: float,
        sci: float,
        budget_scale: float = 1.0,
    ) -> CandidateDensityDecision:

        target = self._target_budget(
            sci
        )

        # V2f trust factor (<= 1): shrinks the whole budget, floor
        # included. Default 1.0 keeps V2b/V2c behaviour unchanged.
        if budget_scale < 1.0:
            target = max(
                1,
                int(round(target * float(budget_scale))),
            )

        base_sensitivity = float(
            np.clip(
                base_sensitivity,
                self.min_sensitivity,
                1.0,
            )
        )

        if (
            self.raw_count_ema is None
            or self.raw_count_ema <= 0.0
        ):
            # First frame:
            # no detector-density history yet.
            effective = base_sensitivity

        else:
            desired_fraction = (
                float(target)
                /
                float(self.raw_count_ema)
            )

            desired_fraction = float(
                np.clip(
                    desired_fraction,
                    self.min_sensitivity,
                    1.0,
                )
            )

            # IMPORTANT:
            # density control is overload protection.
            # It may suppress the base policy,
            # but may NEVER make it more permissive.
            effective = min(
                base_sensitivity,
                desired_fraction,
            )

        low = 1.0 - effective

        high = min(
            0.97,
            low + 0.18,
        )

        new = min(
            0.995,
            high + 0.05,
        )

        return CandidateDensityDecision(
            target_candidates=target,
            raw_count_ema=(
                None
                if self.raw_count_ema is None
                else float(self.raw_count_ema)
            ),
            sensitivity=effective,
            low_threshold=low,
            high_threshold=high,
            new_track_threshold=new,
        )

    def observe(
        self,
        raw_count: int,
    ):
        """
        Frame t count affects t+1 onward only.
        """

        value = float(
            max(
                0,
                int(raw_count),
            )
        )

        if self.raw_count_ema is None:
            self.raw_count_ema = value

        else:
            self.raw_count_ema = (
                self.raw_decay
                * self.raw_count_ema
                +
                (1.0 - self.raw_decay)
                * value
            )
