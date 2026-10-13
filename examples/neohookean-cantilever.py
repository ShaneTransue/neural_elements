"""Neo-Hookean material, cantilever scene: clamped at one end, bending under its own weight.

The material: compressible Neo-Hookean; softens in tension and stiffens in compression.
The model: Neural Elements, nonlinear formulation (the material's trained network, from models/nonlinear/).

    python examples/neohookean-cantilever.py

Other force models: "fem" (finite elements of the same material), "mass_spring".
Other devices: Simulation(..., device="cuda") runs on an NVIDIA GPU (single precision), refine=2 has 8x the elements.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the library is one folder up

from gui import show                                   # noqa: E402
from neural_elements import Material, Simulation       # noqa: E402

sim = Simulation("cantilever", model="ne_nonlinear", material=Material("neohookean"))
show(sim, title="Neo-Hookean material, cantilever")
