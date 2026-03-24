from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GaitScheduler:
    mode: str = "stand"
    cycle_time: float = 0.50
    duty_factor: float = 0.50
    horizon: int = 15
    dt_mpc: float = 0.02

    def __post_init__(self):
        self.trot_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)

    def phase(self, t: float) -> float:
        if self.mode == "stand":
            return 0.0
        return (t / self.cycle_time) % 1.0

    def contact_now(self, t: float) -> np.ndarray:
        if self.mode == "stand":
            return np.ones(4, dtype=int)

        base_phase = self.phase(t)
        out = np.zeros(4, dtype=int)
        for i in range(4):
            leg_phase = (base_phase + float(self.trot_offsets[i])) % 1.0
            out[i] = 1 if leg_phase < self.duty_factor else 0
        return out

    def swing_phase_now(self, t: float) -> np.ndarray:
        if self.mode == "stand":
            return np.zeros(4, dtype=float)

        base_phase = self.phase(t)
        out = np.zeros(4, dtype=float)
        for i in range(4):
            leg_phase = (base_phase + float(self.trot_offsets[i])) % 1.0
            if leg_phase < self.duty_factor:
                out[i] = 0.0
            else:
                out[i] = (leg_phase - self.duty_factor) / max(1e-6, 1.0 - self.duty_factor)
        return out

    def mpc_contact_table(self, t: float) -> np.ndarray:
        table = np.zeros((self.horizon, 4), dtype=int)
        for k in range(self.horizon):
            tk = t + k * self.dt_mpc
            table[k, :] = self.contact_now(tk)
        return table

    def current_stance_time(self) -> float:
        if self.mode == "stand":
            return 10.0 * self.dt_mpc
        return self.duty_factor * self.cycle_time

    def current_swing_time(self) -> float:
        if self.mode == "stand":
            return 0.0
        return (1.0 - self.duty_factor) * self.cycle_time
