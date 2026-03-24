from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def cubic_interp(p0: np.ndarray, pf: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    h = 3 * s**2 - 2 * s**3
    return (1 - h) * p0 + h * pf


def cubic_interp_vel(p0: np.ndarray, pf: np.ndarray, s: float, T: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    T = max(1e-6, float(T))
    dh = (6 * s - 6 * s**2) / T
    return dh * (pf - p0)


@dataclass
class FootSwingTrajectory:
    height: float = 0.08
    p0: np.ndarray | None = None
    pf: np.ndarray | None = None

    def set_start(self, p: np.ndarray) -> None:
        self.p0 = np.array(p, dtype=float)

    def set_goal(self, p: np.ndarray) -> None:
        self.pf = np.array(p, dtype=float)

    def sample(self, phase: float, swing_time: float):
        if self.p0 is None or self.pf is None:
            raise RuntimeError("Swing trajectory endpoints are not initialized.")

        phase = float(np.clip(phase, 0.0, 1.0))
        p = cubic_interp(self.p0, self.pf, phase)
        v = cubic_interp_vel(self.p0, self.pf, phase, swing_time)

        if phase < 0.5:
            z_phase = phase / 0.5
            z = self.p0[2] + self.height * (3 * z_phase**2 - 2 * z_phase**3)
            zd = self.height * (6 * z_phase - 6 * z_phase**2) / max(1e-6, 0.5 * swing_time)
        else:
            z_phase = (phase - 0.5) / 0.5
            z = self.pf[2] + self.height * (1 - (3 * z_phase**2 - 2 * z_phase**3))
            zd = -self.height * (6 * z_phase - 6 * z_phase**2) / max(1e-6, 0.5 * swing_time)

        p = p.copy()
        v = v.copy()
        p[2] = z
        v[2] = zd
        return p, v
