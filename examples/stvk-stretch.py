"""St. Venant-Kirchhoff material, stretch scene: one end pulled out by 32% of the length and back.

The material: linear elasticity on the Green strain; stiffens in tension and softens in compression.
The model: Neural Elements, nonlinear formulation (the material's trained network, from models/nonlinear/).

    python examples/stvk-stretch.py

Other force models: "fem" (finite elements of the same material), "mass_spring".
Other devices: Simulation(..., device="cuda") runs on an NVIDIA GPU (single precision), refine=2 has 8x the elements.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the library is one folder up

from gui import show                                   # noqa: E402
from neural_elements import Material, Simulation       # noqa: E402

sim = Simulation("stretch", model="ne_nonlinear", material=Material("stvk"))
show(sim, title="St. Venant-Kirchhoff material, stretch")
