"""Neural Elements: the interactive application (OpenGL).

    python viewer.py

Choose a scene, a force model (Neural Elements in the nonlinear or the linear formulation, FEM, mass-spring) and a
material (linear, Neo-Hookean, or the blend between them), then Start. Material values change live; the FEM of the
same material can be shown alongside as a ghost. Runs on the CPU, or on a CUDA GPU at up to 64x the elements.
Mouse: left-drag rotate, right-drag pan, wheel zoom, double-click reframe. Keys: Space start/pause, S step, R reset.
"""
from gui.window import main

if __name__ == "__main__":
    main()
