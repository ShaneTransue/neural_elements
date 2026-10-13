"""Neural Elements: finite elements whose internal force response is learned from a reference element.

    from neural_elements import Material, Simulation

    sim = Simulation("stretch", model="ne_nonlinear", material=Material("neohookean"))
    sim.run(sim.scene.loading_frames)

Modules
    materials     the material definition (model, E, nu, kappa) and the material models' stresses
    linear        Neural Elements, linear formulation (single-node load cases)
    nonlinear     Neural Elements, nonlinear formulation (one deformation load case, learned as an energy)
    fem           finite elements: linear corotational, Neo-Hookean, blend
    mass_spring   springs on the mesh edges
    training      training of both networks on the reference element
    solver        implicit Newmark, backward Euler, explicit Euler; matrix-free conjugate gradients
    scenes        beam meshes and the four test scenes
    simulation    scene + force model + integrator
    geometry      linear tetrahedra as element-wise tensor operations
"""
from .materials import Material
from .scenes import SCENES
from .simulation import FORCE_MODELS, Simulation, load_network, make_model

__all__ = ["Material", "SCENES", "FORCE_MODELS", "Simulation", "load_network", "make_model"]
