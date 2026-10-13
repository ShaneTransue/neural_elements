"""One 3D view of a running simulation, for small scripts: show(simulation).

    from gui import show
    show(Simulation("twist", material=Material("neohookean")))

The simulation is stepped in real time (one step per frame, at the scene's time step) and drawn colored by von Mises
stress, with the frame, the time and the scene's main measurement on top. The scene starts over at its end.
Mouse: left-drag rotate, right-drag pan, wheel zoom, double-click reframe. Keys: Space pause, R restart.

The calling script also accepts  --screenshot FILE [--frames N]: run N frames (default 120) without showing a window,
save the view to FILE and exit (for pictures and tests).
"""
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from neural_elements.scenes import boundary_faces

from .window import Viewport, set_default_format

MEASUREMENT = {"cantilever": ("tip", "tip deflection"), "stretch": ("force", "end force"),
               "compress": ("force", "end force"), "twist": ("torque", "end torque")}


def show(sim, title=None):
    set_default_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    view = Viewport()
    view.setWindowTitle(title or f"Neural Elements: {sim.scene.name}")
    view.resize(1000, 620)
    faces, rest, fixed = boundary_faces(sim.elems), sim.rest.numpy(), sim.boundary.fixed.numpy()
    key, label = MEASUREMENT.get(sim.scene.name, ("tip", "tip deflection"))
    state = dict(paused=False)

    def draw():
        x = sim.positions()
        L = sim.last
        view.overlay = (f"{title or sim.scene.name}\nframe {sim.frame}   t = {sim.time:.2f} s" +
                        (f"   |   {label} {L.get(key, 0.0):.4g}   |   volume {100 * (L['volume'] - 1):+.1f}%" if L else ""))
        view.show_snapshot(dict(x=x, faces=faces, rest=rest, fixed=fixed, vm=sim.von_mises(),
                                disp=np.linalg.norm(x - rest, axis=1)))

    def tick():
        if state["paused"]:
            return
        if sim.frame >= sim.scene.frames:                   # start the scene over
            sim.reset()
        t0 = time.perf_counter()
        sim.step()
        draw()
        timer.setInterval(max(0, int(1000 * (sim.dt - (time.perf_counter() - t0)))))

    def restart():
        sim.reset()
        draw()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    QtWidgets.QShortcut(QtGui.QKeySequence("Space"), view, activated=lambda: state.update(paused=not state["paused"]))
    QtWidgets.QShortcut(QtGui.QKeySequence("R"), view, activated=restart)
    draw()                                                   # (frames the camera on the beam)
    view.camera.dist *= 0.72                                 # one view only: come closer

    if "--screenshot" in sys.argv:                           # run without a window, save the view, exit
        arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        path, frames = arg("--screenshot", "view.png"), int(arg("--frames", "120"))
        view.setAttribute(Qt.WA_DontShowOnScreen)
        view.show()
        for _ in range(frames):
            sim.step()
        draw()
        QtCore.QTimer.singleShot(300, lambda: (view.grabFramebuffer().save(path), app.quit()))
    else:
        view.show()
        timer.start(0)
    app.exec_()
