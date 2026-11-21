"""Opt-in ecology-v1 pressures; the historical Cfg and protocols stay unchanged."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import ClassVar

import jax
import jax.numpy as jnp

SHELTER_STREAM_TAG = 0x45434F31


@dataclass(frozen=True)
class EcologyV1:
    movement_energy_cost: float = 0.0
    seasonal_energy_cost: float = 0.0
    version: ClassVar[str] = "ecology-v1"

    def __post_init__(self):
        for name in ("movement_energy_cost", "seasonal_energy_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


def make_shelter(wkey, c):
    """Same-size toroidal shelter at a fresh location, without consuming world RNG."""
    anchor = jax.random.randint(jax.random.fold_in(wkey, SHELTER_STREAM_TAG), (2,), 0, c.G)
    width = max(1, c.G // 4)
    row, column = jnp.arange(c.G)[:, None], jnp.arange(c.G)[None, :]
    return ((row - anchor[0]) % c.G < width) & ((column - anchor[1]) % c.G < width)


def apply_costs(energy, action, alive, step, half_life, ecology, *, exposed=True):
    """Charge living agents before energy clipping; zero costs add no array operations."""
    if ecology is None:
        return energy
    if not isinstance(ecology, EcologyV1):
        raise ValueError("ecology must be an EcologyV1 instance or None")
    if ecology.movement_energy_cost:
        energy = energy - ecology.movement_energy_cost * (action != 0) * alive
    if ecology.seasonal_energy_cost:
        energy = energy - ecology.seasonal_energy_cost * (step >= half_life) * exposed * alive
    return energy
