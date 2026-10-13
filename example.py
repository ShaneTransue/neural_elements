"""Neural Elements: terminal example (no GUI).

    python example.py                  about a minute on a laptop CPU
    python example.py --device cuda    on an NVIDIA GPU (single precision)

It first prints the expected output, then runs the same simulations and prints what it gets, row by row.

What it shows: Neural Elements, whose internal forces come from small networks trained only on one reference element,
reproduce the finite-element model of the same material in whole simulations:
    1. a cantilever bending under its own weight (linear material): FEM, both Neural Element formulations, and a
       mass-spring system for contrast;
    2. a beam stretched to 32% strain, with linear, Neo-Hookean, St. Venant-Kirchhoff and blended materials: their
       peak forces differ by up to about 90%, and the nonlinear formulation follows each one.

Requirements: torch and numpy, and the trained networks in models/ (made by the scripts in training/).
"""
import argparse
import time

from neural_elements import Material, Simulation

# Recorded with the trained networks in models/ (CPU, double precision). Retrained networks give values that differ
# slightly in the Neural Element rows; FEM and mass-spring rows do not depend on them.
EXPECTED = {
    "cantilever": {"fem": -1.9281, "ne_linear": -1.9272, "ne_nonlinear": -1.9254, "mass_spring": -1.7599},
    "stretch": {("linear", "fem"): 31542.0, ("linear", "ne_linear"): 31551.1, ("linear", "ne_nonlinear"): 31655.4,
                ("neohookean", "fem"): 25739.2, ("neohookean", "ne_nonlinear"): 25723.6,
                ("stvk", "fem"): 48702.4, ("stvk", "ne_nonlinear"): 48787.2,
                ("blend", "fem"): 28694.6, ("blend", "ne_nonlinear"): 28738.5},
}
TOLERANCE = {"cpu": 0.05, "cuda": 0.5}     # percent: double precision on the CPU, single precision on the GPU
NAMES = {"fem": "FEM", "ne_linear": "Neural Elements, linear formulation",
         "ne_nonlinear": "Neural Elements, nonlinear formulation", "mass_spring": "Mass-spring"}
MATERIAL_NAMES = {"linear": "linear", "neohookean": "Neo-Hookean", "stvk": "St. V.-Kirchhoff", "blend": "blend 0.5"}


def cantilever_tip(model, device):
    """Tip deflection of the cantilever after 1 s (60 frames), linear material."""
    sim = Simulation("cantilever", model, Material("linear"), device=device)
    sim.run(60)
    return sim.history["tip"][-1]


def stretch_peak_force(model, material, device):
    """Peak end force while the beam is stretched to 32% strain (the loading half of the scene)."""
    sim = Simulation("stretch", model, Material(material, kappa=0.5), device=device)
    sim.run(sim.scene.loading_frames)
    return max(abs(f) for f in sim.history["force"])


def table(results, check=None, tol=0.05):
    """Print both parts; with `check` (the expected values) each row also says whether it matches."""
    good = True

    def row(label, value, ref, expected, fmt):
        nonlocal good
        vs = f"{100 * (value / ref - 1):+7.2f}%" if value != ref else ""
        line = f"     {label:56s} {value:{fmt}}  {vs:>8s}"
        if check is not None:
            match = expected is not None and abs(value / expected - 1) * 100 <= tol
            good &= match
            line += "    ok" if match else f"    DIFFERS (expected {expected})"
        print(line)

    print("1. Cantilever bending under its own weight, linear material: tip deflection after 1 s")
    print(f"     {'model':56s} {'tip':>10s}  {'vs FEM':>8s}")
    c = results["cantilever"]
    for model, v in c.items():
        row(NAMES[model], v, c["fem"], check and check["cantilever"][model], "10.4f")
    print("\n2. Beam stretched to 32% strain: peak end force")
    print(f"     {'material          model':56s} {'force':>10s}  {'vs FEM':>8s}")
    s = results["stretch"]
    for (material, model), v in s.items():
        row(f"{MATERIAL_NAMES[material]:17s} {NAMES[model]}", v, s[(material, "fem")],
            check and check["stretch"][(material, model)], "10.1f")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    a = ap.parse_args()
    print("Neural Elements - terminal example\n")
    print("Expected output\n---------------")
    table(EXPECTED)
    precision = "double" if a.device == "cpu" else "single"
    print(f"\nRunning on {a.device.upper()} in {precision} precision\n{'-' * 40}")
    t0 = time.time()
    results = {"cantilever": {m: cantilever_tip(m, a.device) for m in EXPECTED["cantilever"]},
               "stretch": {k: stretch_peak_force(k[1], k[0], a.device) for k in EXPECTED["stretch"]}}
    tol = TOLERANCE["cpu" if a.device == "cpu" else "cuda"]
    good = table(results, EXPECTED, tol)
    n = len(results["cantilever"]) + len(results["stretch"])
    print(f"\n{'All' if good else 'Not all'} {n} values match the expected output (within {tol}%); "
          f"{time.time() - t0:.0f} s.")
    return results


if __name__ == "__main__":
    main()
