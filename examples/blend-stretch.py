"""Blended material, stretch scene: one end pulled out by 32% of the length and back.

The material: halfway between linear (kappa = 0) and Neo-Hookean (kappa = 1); kappa is a network input.
The model: Neural Elements, nonlinear formulation (the material's trained network, from models/nonlinear/).

    python examples/blend-stretch.py

Other force models: "fem" (finite elements of the same material), "mass_spring".
Other devices: Simulation(..., device="cuda") runs on an NVIDIA GPU (single precision), refine=2 has 8x the elements.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the library is one folder up

from gui import show                                   # noqa: E402
from neural_elements import Material, Simulation       # noqa: E402

sim = Simulation("stretch", model="ne_nonlinear", material=Material("blend", kappa=0.5))
show(sim, title="Blended material, stretch")
