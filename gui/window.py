"""The interactive application: a scene simulated live, drawn with OpenGL, with every choice of the library on a panel.

    python viewer.py

The simulation runs in its own thread (Worker) and sends a snapshot after every frame; the window draws the latest
snapshot (Viewport, through gui/renderer.py) and a time plot of the scene's main measurement (Plot).
Mouse: left-drag rotate, right- or middle-drag pan, wheel zoom, double-click reframe.
Keys:  Space start/pause, S single step, R reset, F reframe.
"""
import math
import os
import sys
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from neural_elements import FORCE_MODELS, SCENES, Material, Simulation
from neural_elements.scenes import boundary_faces
from neural_elements.solver import INTEGRATORS

from .renderer import Camera, Renderer, colormap_table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERIALS = [("linear", "Linear"), ("neohookean", "Neo-Hookean"), ("stvk", "St. Venant-Kirchhoff"),
             ("blend", "Blend (kappa)")]
COLOR_MODES = ["von Mises stress", "displacement", "difference to FEM", "plain"]
PLOT = {"cantilever": ("tip", "tip displacement y"), "stretch": ("force", "end force"),
        "compress": ("force", "end force"), "twist": ("torque", "end torque")}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PRECISION = {"double": torch.float64, "single": torch.float32}
NU_RANGE = (0.0, 0.45)      # trained range of the networks
DIFF_RANGE = 10.0           # color range of the difference to FEM: 0 - 10% of the FEM's largest displacement
ABOUT = {"ne_nonlinear": "One deformation load case per element; the network gives the reference element's energy.",
         "ne_linear": "Twelve single-node load cases per element (the dissertation's formulation); linear only.",
         "fem": "Finite elements of the material: linear corotational, Neo-Hookean, St. Venant-Kirchhoff, blend.",
         "mass_spring": "Springs on the mesh edges (no Poisson's ratio)."}


def set_default_format():
    """OpenGL 4.1 core profile with 4x multisampling, for every OpenGL surface of the application."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


# ---------------------------------------------------------------------------------------------------------------------
# simulation thread
# ---------------------------------------------------------------------------------------------------------------------
class Worker(QtCore.QObject):
    snapshot = pyqtSignal(object)
    message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sim = self.ref = None
        self.running, self.realtime = False, True
        self.cfg = {}

    @pyqtSlot()
    def start(self):
        torch.set_num_threads(4)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(0)

    def make_sim(self, cfg, model):
        mat = Material(cfg["material"], cfg["E"], cfg["nu"], cfg["kappa"])
        return Simulation(cfg["scene"], model, mat, cfg["integrator"], refine=cfg["refine"], device=cfg["device"],
                          dtype=PRECISION[cfg["precision"]])

    @pyqtSlot(object)
    def configure(self, cfg):
        self.cfg = dict(cfg)
        self.sim = self.ref = None
        try:
            self.sim = self.make_sim(cfg, cfg["model"])
            self.ref = self.make_sim(cfg, "fem") if cfg["compare"] and cfg["model"] != "fem" else None
        except Exception as exc:  # noqa: BLE001  (shown in the status bar, e.g. a missing trained network)
            self.sim = None
            self.message.emit(f"could not build the simulation: {exc}")
            return
        self._faces = boundary_faces(self.sim.elems)
        self.apply_material(cfg)
        self._emit()

    @pyqtSlot(bool)
    def set_running(self, on):
        self.running = on

    @pyqtSlot(bool)
    def set_realtime(self, on):
        self.realtime = on

    @pyqtSlot()
    def step_once(self):
        if self.sim is not None:
            self._advance()

    @pyqtSlot()
    def reset(self):
        for s in (self.sim, self.ref):
            if s is not None:
                s.reset()
        if self.sim is not None:
            self._emit()

    @pyqtSlot(object)
    def set_material(self, cfg):
        self.cfg.update(cfg)
        self.apply_material(self.cfg)
        if self.sim is not None:
            self._emit()

    def apply_material(self, cfg):
        """E, nu (and kappa for the blend) of every element; applied without rebuilding anything."""
        for s in (self.sim, self.ref):
            if s is not None:
                s.set_material(E=cfg["E"], nu=cfg["nu"],
                               kappa=cfg["kappa"] if s.material.model == "blend" else None)

    def _advance(self):
        t0 = time.perf_counter()
        self.sim.step()
        if self.ref is not None:
            self.ref.step()
        self._emit()
        if not np.isfinite(self.sim.last.get("tip", 0.0)):
            self.running = False
            self.message.emit("the simulation diverged (press Reset)")
        return time.perf_counter() - t0

    def _tick(self):
        if not self.running or self.sim is None:
            self.timer.setInterval(20)
            return
        if self.sim.frame >= 4 * self.sim.scene.frames:
            self.running = False
            self.message.emit("end of the run (press Reset)")
            return
        spent = self._advance()
        self.timer.setInterval(max(0, int(1000 * (self.sim.dt - spent))) if self.realtime else 0)

    def _emit(self):
        s = self.sim
        x = s.positions()
        keep = ("time", "tip", "force", "torque")
        snap = dict(x=x, faces=self._faces, rest=s.rest.numpy(), frame=s.frame, time=s.time, last=dict(s.last),
                    vm=s.von_mises(), disp=np.linalg.norm(x - s.rest.numpy(), axis=1), fixed=s.boundary.fixed.numpy(),
                    history={k: list(v) for k, v in s.history.items() if k in keep},
                    info=f"{s.device.type.upper()}  |  {s.model.Ne:,} elements  |  "
                         f"{'double' if s.dtype == torch.float64 else 'single'} precision")
        if self.ref is not None:
            xr = self.ref.positions()
            snap.update(xref=xr, diff=np.linalg.norm(x - xr, axis=1),
                        ref_history={k: list(v) for k, v in self.ref.history.items() if k in keep})
        self.snapshot.emit(snap)


# ---------------------------------------------------------------------------------------------------------------------
# 3D view
# ---------------------------------------------------------------------------------------------------------------------
class Viewport(QtWidgets.QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.StrongFocus)
        self.renderer, self.camera = Renderer(), Camera()
        self.snap, self.faces = None, None
        self.mode, self.wire, self.show_fixed = COLOR_MODES[0], False, True
        self.vmax, self.overlay = {}, ""
        self.tables = {name: colormap_table(name) for name in ("viridis", "magma_r")}
        self._drag = None

    def initializeGL(self):
        vp = QtGui.QOpenGLVersionProfile()
        vp.setVersion(4, 1)
        vp.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        gl = self.context().versionFunctions(vp)
        if gl is None:
            raise RuntimeError("OpenGL 4.1 is not available")
        gl.initializeOpenGLFunctions()
        self.renderer.initialize(gl)

    def show_snapshot(self, s):
        self.snap = s
        if s["faces"] is not self.faces:          # a new simulation: take its mesh and frame it
            self.faces = s["faces"]
            self.vmax = {}
            self.camera.frame(s["x"])
            self.pending_mesh = True
        self.update()

    def scalars(self):
        s = self.snap
        if self.mode == "von Mises stress":
            return s["vm"], "viridis"
        if self.mode == "displacement":
            return s["disp"], "viridis"
        if self.mode == "difference to FEM" and "diff" in s:
            # relative to how far the FEM has moved, so that a difference reads as small when it is small
            motion = float(np.linalg.norm(s["xref"] - s["rest"], axis=1).max())
            return 100 * s["diff"] / max(motion, 1e-12), "magma_r"
        return None, "viridis"

    def paintGL(self):
        r = self.renderer
        vals, cmap = (None, "viridis")
        if self.snap is not None:
            if getattr(self, "pending_mesh", False):
                r.set_mesh(self.faces, self.snap["fixed"], self.snap["rest"])
                self.pending_mesh = False
            vals, cmap = self.scalars()
            scaled = None
            if vals is not None:
                if self.mode == "difference to FEM":
                    self.vmax[self.mode] = DIFF_RANGE                       # fixed range, in percent
                else:
                    hi = float(np.percentile(vals, 99)) if len(vals) else 1.0
                    old = self.vmax.get(self.mode, 0.0)
                    self.vmax[self.mode] = max(hi, old * 0.995, 1e-9)      # slowly decaying color range
                scaled = vals / self.vmax[self.mode]
            r.update(self.snap["x"], scaled, self.snap.get("xref"))
        ratio = self.devicePixelRatioF()
        r.draw(self.camera, int(self.width() * ratio), int(self.height() * ratio), cmap, self.wire, self.show_fixed,
               7.0 * ratio)
        self.paint_overlay(vals, cmap)

    # overlay (QPainter on top of the OpenGL image) -------------------------------------------------------------------
    def paint_overlay(self, vals, cmap):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setPen(QtGui.QColor(30, 30, 30))
        qp.setFont(QtGui.QFont("Segoe UI", 9))
        if self.snap is None:
            qp.drawText(self.rect(), Qt.AlignCenter, "no simulation")
            qp.end()
            return
        qp.drawText(QtCore.QRectF(10, 8, 700, 120), Qt.AlignLeft | Qt.AlignTop, self.overlay)
        _, axes = self.camera.basis()
        o = QtCore.QPointF(45, self.height() - 45)
        for axis, col, lab in ((0, (200, 40, 40), "x"), (1, (40, 160, 40), "y"), (2, (40, 70, 220), "z")):
            v = axes[:, axis]
            p = QtCore.QPointF(o.x() + 28 * v[0], o.y() - 28 * v[1])
            qp.setPen(QtGui.QPen(QtGui.QColor(*col), 2))
            qp.drawLine(o, p)
            qp.drawText(p + QtCore.QPointF(3, 3), lab)
        if vals is not None:
            w, h, x0, y0 = 14, 180, self.width() - 95, 30
            L = self.tables[cmap]
            for i in range(h):
                c = L[int(255 * (1 - i / (h - 1)))]
                qp.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
                qp.drawLine(x0, y0 + i, x0 + w, y0 + i)
            qp.setPen(QtGui.QColor(40, 40, 40))
            qp.drawRect(x0, y0, w, h)
            diff = self.mode == "difference to FEM"
            unit = "%" if diff else ""
            qp.drawText(x0 + w + 4, y0 + 8, f"{self.vmax[self.mode]:.3g}{unit}")
            qp.drawText(x0 + w + 4, y0 + h, f"0{unit}")
            captions = ("difference to FEM", "in % of its largest displacement") if diff else (self.mode,)
            lh = qp.fontMetrics().height() + 3
            for k, text in enumerate(captions):
                qp.drawText(QtCore.QRectF(x0 - 190, y0 + h + 6 + lh * k, 260, lh), Qt.AlignRight, text)
        qp.end()

    # mouse -----------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, e):
        self._drag = (e.button(), e.pos())

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        b, p0 = self._drag
        dx, dy = e.x() - p0.x(), e.y() - p0.y()
        cam = self.camera
        if b == Qt.LeftButton:
            cam.yaw -= dx * 0.01
            cam.pitch = min(max(cam.pitch + dy * 0.01, -1.5), 1.5)
        else:
            _, axes = cam.basis()
            cam.target = cam.target + (-dx * axes[0] + dy * axes[1]) * cam.dist * 0.0015
        self._drag = (b, e.pos())
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.reframe()

    def wheelEvent(self, e):
        self.camera.dist = min(max(self.camera.dist * 0.9 ** (e.angleDelta().y() / 120), 1.0), 500.0)
        self.update()

    def reframe(self):
        if self.snap is not None:
            self.camera.frame(self.snap["x"])
            self.update()


# ---------------------------------------------------------------------------------------------------------------------
# time plot
# ---------------------------------------------------------------------------------------------------------------------
class Plot(QtWidgets.QWidget):
    """Time plot of the scene's main measurement (blue: the chosen model, black: the FEM ghost), on a grid at round
    values of time and of the measurement."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.series, self.ref, self.label = [], [], ""

    def set(self, t, y, tr, yr, label):
        self.series, self.ref, self.label = list(zip(t, y)), list(zip(tr, yr)), label
        self.update()

    @staticmethod
    def ticks(lo, hi, n):
        """Round tick values (steps of 1, 2 or 5 times a power of ten) covering lo..hi, about n of them."""
        raw = (hi - lo) / max(n, 1)
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
        return step, [k * step for k in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
        qp.setFont(QtGui.QFont("Segoe UI", 8))
        left, right, top, bottom = 56, 14, 24, 32
        W, H = self.width() - left - right, self.height() - top - bottom
        qp.setPen(QtGui.QColor(60, 60, 60))
        qp.drawText(left + 4, 15, self.label)
        pts = self.series + self.ref
        t1 = max([p[0] for p in pts] + [1.0])
        ys = [p[1] for p in pts] or [0.0, 1.0]
        y0, y1 = min(min(ys), 0.0), max(max(ys), 0.0)
        if y1 - y0 < 1e-12:
            y0, y1 = y0 - 0.5, y1 + 0.5
        pad = 0.05 * (y1 - y0)
        y0, y1 = y0 - (pad if y0 < 0 else 0), y1 + (pad if y1 > 0 else 0)
        X = lambda v: left + W * v / t1
        Y = lambda v: top + H * (1 - (v - y0) / (y1 - y0))
        # grid at round values, labels on the left (values) and below (time)
        grid, label = QtGui.QPen(QtGui.QColor(232, 234, 238), 1), QtGui.QColor(110, 110, 110)
        step, values = self.ticks(y0, y1, max(2, H // 28))
        for v in values:
            qp.setPen(grid)
            qp.drawLine(QtCore.QPointF(left, Y(v)), QtCore.QPointF(left + W, Y(v)))
            qp.setPen(label)
            qp.drawText(QtCore.QRectF(0, Y(v) - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        _, times = self.ticks(0.0, t1, max(2, W // 70))
        for v in times:
            qp.setPen(grid)
            qp.drawLine(QtCore.QPointF(X(v), top), QtCore.QPointF(X(v), top + H))
            if X(v) < left + W - 55:                                   # leave room for the axis name
                qp.setPen(label)
                qp.drawText(QtCore.QRectF(X(v) - 30, top + H + 3, 60, 16), Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")
        qp.setPen(label)
        qp.drawText(QtCore.QRectF(left + W - 60, top + H + 3, 60, 16), Qt.AlignRight | Qt.AlignTop, "t [s]")
        qp.setPen(QtGui.QPen(QtGui.QColor(170, 172, 178), 1))
        qp.drawLine(QtCore.QPointF(left, Y(0)), QtCore.QPointF(left + W, Y(0)))      # zero line
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(QtCore.QRectF(left, top, W, H))
        for data, col in ((self.ref, QtGui.QColor(30, 30, 30)), (self.series, QtGui.QColor(27, 67, 251))):
            if len(data) > 1:
                qp.setPen(QtGui.QPen(col, 1.5))
                qp.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(X(a), Y(b)) for a, b in data]))


# ---------------------------------------------------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------------------------------------------------
class Main(QtWidgets.QMainWindow):
    configure = pyqtSignal(object)
    set_running = pyqtSignal(bool)
    set_realtime = pyqtSignal(bool)
    step_once = pyqtSignal()
    reset_sim = pyqtSignal()
    set_material = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Elements")
        self.view, self.plot = Viewport(), Plot()
        self.running = False

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(410)
        form = QtWidgets.QVBoxLayout(panel)
        grid = QtWidgets.QFormLayout()
        combo = lambda items: self._combo(items)
        self.scene = combo([(k, k) for k in SCENES])
        self.model = combo(list(FORCE_MODELS.items()))
        self.material = combo(MATERIALS)
        self.integrator = combo([(k, v.name) for k, v in INTEGRATORS.items()])
        self.resolution = combo([(r, f"{r}x") for r in (1, 2, 3, 4)])
        self.device = combo([(d, d.upper() + (f"  ({torch.cuda.get_device_name(0)})" if d == "cuda" else ""))
                             for d in DEVICES])
        self.precision = combo([(p, p) for p in PRECISION])
        for label, w in (("Scene", self.scene), ("Model", self.model), ("Material", self.material),
                         ("Integrator", self.integrator), ("Resolution", self.resolution), ("Device", self.device),
                         ("Precision", self.precision)):
            grid.addRow(label, w)
        form.addLayout(grid)
        self.desc = QtWidgets.QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: #555")
        form.addWidget(self.desc)

        row = QtWidgets.QHBoxLayout()
        self.btn_run, self.btn_step, self.btn_reset = (QtWidgets.QPushButton(t) for t in ("Start", "Step", "Reset"))
        for b in (self.btn_run, self.btn_step, self.btn_reset):
            row.addWidget(b)
        form.addLayout(row)

        box = QtWidgets.QGroupBox("Display")
        bl = QtWidgets.QFormLayout(box)
        self.color = QtWidgets.QComboBox()
        self.color.addItems(COLOR_MODES)
        bl.addRow("Color", self.color)
        self.cb_compare = QtWidgets.QCheckBox("FEM of the same material (ghost)")
        self.cb_wire = QtWidgets.QCheckBox("Wireframe")
        self.cb_fixed = QtWidgets.QCheckBox("Constrained nodes")
        self.cb_realtime = QtWidgets.QCheckBox("Real-time pacing")
        self.cb_fixed.setChecked(True)
        self.cb_realtime.setChecked(True)
        for cb in (self.cb_compare, self.cb_wire, self.cb_fixed, self.cb_realtime):
            bl.addRow(cb)
        form.addWidget(box)

        box = QtWidgets.QGroupBox("Material values (applied live)")
        ml = QtWidgets.QFormLayout(box)
        self.sE, self.snu, self.skappa = (QtWidgets.QSlider(Qt.Horizontal) for _ in range(3))
        self.sE.setRange(500, 40000)
        self.snu.setRange(0, 490)
        self.skappa.setRange(0, 100)
        self.skappa.setValue(50)
        self.lE, self.lnu, self.lkappa = QtWidgets.QLabel(), QtWidgets.QLabel(), QtWidgets.QLabel()
        ml.addRow(self.lE, self.sE)
        ml.addRow(self.lnu, self.snu)
        ml.addRow(self.lkappa, self.skappa)
        self.btn_mat = QtWidgets.QPushButton("Restore scene values")
        ml.addRow(self.btn_mat)
        form.addWidget(box)

        self.stats = QtWidgets.QLabel()
        self.stats.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt")
        self.stats.setAlignment(Qt.AlignTop)
        form.addWidget(self.stats)
        self.btn_shot = QtWidgets.QPushButton("Save screenshot")
        form.addWidget(self.btn_shot)
        form.addStretch(1)

        left = QtWidgets.QSplitter(Qt.Vertical)
        left.addWidget(self.view)
        left.addWidget(self.plot)
        left.setSizes([640, 170])
        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(left, 1)
        lay.addWidget(panel)
        self.setCentralWidget(central)
        self.statusBar()

        # the simulation thread
        self.thread = QtCore.QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        for sig, slot in ((self.configure, self.worker.configure), (self.set_running, self.worker.set_running),
                          (self.set_realtime, self.worker.set_realtime), (self.step_once, self.worker.step_once),
                          (self.reset_sim, self.worker.reset), (self.set_material, self.worker.set_material)):
            sig.connect(slot)
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.message.connect(self.statusBar().showMessage)
        self.thread.start()

        # controls
        for w in (self.scene, self.model, self.material, self.integrator, self.resolution, self.device, self.precision):
            w.currentIndexChanged.connect(self.rebuild)
        self.cb_compare.toggled.connect(self.rebuild)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(lambda: self.step_once.emit())
        self.btn_reset.clicked.connect(lambda: self.reset_sim.emit())
        self.color.currentTextChanged.connect(self.on_color)
        self.cb_wire.toggled.connect(lambda v: (setattr(self.view, "wire", v), self.view.update()))
        self.cb_fixed.toggled.connect(lambda v: (setattr(self.view, "show_fixed", v), self.view.update()))
        self.cb_realtime.toggled.connect(lambda v: self.set_realtime.emit(v))
        for w in (self.sE, self.snu, self.skappa):
            w.valueChanged.connect(self.on_material)
        self.btn_mat.clicked.connect(self.restore_material)
        self.btn_shot.clicked.connect(self.screenshot)
        for key, fn in (("Space", self.toggle_run), ("S", lambda: self.step_once.emit()),
                        ("R", lambda: self.reset_sim.emit()), ("F", self.view.reframe)):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        self.device.setCurrentIndex(self.device.findData("cuda") if "cuda" in DEVICES else 0)
        self.precision.setCurrentIndex(self.precision.findData("single" if "cuda" in DEVICES else "double"))
        self.resize(1360, 880)
        self.rebuild()

    @staticmethod
    def _combo(items):
        c = QtWidgets.QComboBox()
        for key, label in items:
            c.addItem(label, key)
        return c

    # actions ---------------------------------------------------------------------------------------------------------
    def current(self):
        return dict(scene=self.scene.currentData(), model=self.model.currentData(),
                    material=self.material.currentData(), integrator=self.integrator.currentData(),
                    refine=self.resolution.currentData(), device=self.device.currentData(),
                    precision=self.precision.currentData(), compare=self.cb_compare.isChecked(),
                    E=float(self.sE.value()), nu=self.snu.value() / 1000.0, kappa=self.skappa.value() / 100.0)

    def rebuild(self):
        sc = SCENES[self.scene.currentData()]
        linear_only = self.model.currentData() == "ne_linear"
        if linear_only and self.material.currentData() != "linear":
            self.material.blockSignals(True)
            self.material.setCurrentIndex(self.material.findData("linear"))
            self.material.blockSignals(False)
            self.statusBar().showMessage("the linear formulation carries the linear material only", 5000)
        self.material.setEnabled(not linear_only)
        self.skappa.setEnabled(self.material.currentData() == "blend")
        cells = sc.cells
        for i in range(self.resolution.count()):
            r = self.resolution.itemData(i)
            self.resolution.setItemText(i, f"{r}x  ({5 * cells[0] * cells[1] * cells[2] * r ** 3:,} elements)")
        self.desc.setText(f"{sc.description}. E = {sc.E:g}, nu = {sc.nu}, dt = 1/{round(1 / sc.dt)} s.\n"
                          f"{ABOUT[self.model.currentData()]}")
        for s, v in ((self.sE, int(sc.E)), (self.snu, int(sc.nu * 1000))):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.update_labels()
        self.running = False
        self.set_running.emit(False)
        self.btn_run.setText("Start")
        self.configure.emit(self.current())

    def toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("Pause" if self.running else "Start")
        self.set_running.emit(self.running)

    def on_color(self, mode):
        self.view.mode = mode
        if mode == "difference to FEM" and not self.cb_compare.isChecked():
            self.statusBar().showMessage("enable the FEM ghost to color by the difference", 4000)
        self.view.update()

    def update_labels(self):
        nu = self.snu.value() / 1000
        neural = self.model.currentData().startswith("ne_")
        outside = neural and not NU_RANGE[0] <= nu <= NU_RANGE[1]
        self.lE.setText(f"E = {self.sE.value()}")
        self.lnu.setText(f"nu = {nu:.3f}")
        self.lnu.setStyleSheet("color: #c05000" if outside else "")
        self.lkappa.setText(f"kappa = {self.skappa.value() / 100:.2f}")
        if outside:
            self.statusBar().showMessage(f"nu outside the networks' trained range ({NU_RANGE[0]:g} - {NU_RANGE[1]:g}); "
                                         "E has no range limit", 4000)

    def on_material(self):
        self.update_labels()
        c = self.current()
        self.set_material.emit({k: c[k] for k in ("E", "nu", "kappa")})

    def restore_material(self):
        sc = SCENES[self.scene.currentData()]
        self.sE.setValue(int(sc.E))
        self.snu.setValue(int(sc.nu * 1000))
        self.skappa.setValue(50)

    def screenshot(self, path=None):
        if not path:
            os.makedirs(os.path.join(HERE, "screenshots"), exist_ok=True)
            path = os.path.join(HERE, "screenshots", time.strftime("view_%Y%m%d_%H%M%S.png"))
        img = self.grab()        # the window; the 3D view is painted in from its own framebuffer (with its overlay)
        qp = QtGui.QPainter(img)
        qp.drawImage(QtCore.QRect(self.view.mapTo(self, QtCore.QPoint(0, 0)), self.view.size()),
                     self.view.grabFramebuffer())
        qp.end()
        img.save(path)
        self.statusBar().showMessage(f"saved {path}", 5000)

    @pyqtSlot(object)
    def on_snapshot(self, s):
        c = self.current()
        L = s["last"]
        models, mats = dict(FORCE_MODELS), dict(MATERIALS)
        mat = mats[c["material"]] if c["model"] != "mass_spring" else "springs"
        self.view.overlay = (f"{c['scene']}  |  {models[c['model']]}  |  {mat}\n"
                             f"frame {s['frame']}   t = {s['time']:.2f} s\n{s['info']}")
        self.view.show_snapshot(s)
        key, label = PLOT[c["scene"]]
        lines = [f"frame        {s['frame']}", f"time         {s['time']:.2f} s"]
        if L:
            lines += [f"step         {L['ms']:.0f} ms",
                      f"Newton its   {L['iterations']} / CG {L['cg_iterations']}" + ("" if L["converged"] else "  (not converged)"),
                      f"{label[:12]:12s} {L.get(key, float('nan')):.4g}",
                      f"volume       {100 * (L['volume'] - 1):+.1f}%", f"inverted     {L['inverted']}"]
            if "diff" in s:
                motion = float(np.sqrt(((s["xref"] - s["rest"]) ** 2).sum(1).mean()))
                if motion > 1e-6:
                    lines.append(f"vs FEM       {100 * float(np.sqrt((s['diff'] ** 2).mean())) / motion:.1f}% (RMS)")
        self.stats.setText("\n".join(lines))
        h, hr = s["history"], s.get("ref_history", {})
        self.plot.set(h.get("time", []), h.get(key, []), hr.get("time", []), hr.get(key, []),
                      f"{label} (blue: {models[c['model']]}" + (", black: FEM)" if hr else ")"))

    def closeEvent(self, e):
        self.set_running.emit(False)
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(e)


def main():
    """Start the application. Options for scripted screenshots (used to make the figures of the documentation):
        --scene NAME --model NAME --material NAME --compare --wire --color MODE --frames N --screenshot FILE"""
    set_default_format()
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    for key, combo in (("--scene", w.scene), ("--model", w.model), ("--material", w.material),
                       ("--device", w.device), ("--precision", w.precision)):
        if arg(key):
            combo.setCurrentIndex(combo.findData(arg(key)))
    if arg("--refine"):
        w.resolution.setCurrentIndex(w.resolution.findData(int(arg("--refine"))))
    w.cb_compare.setChecked("--compare" in sys.argv)
    w.cb_wire.setChecked("--wire" in sys.argv)
    if arg("--color"):
        w.color.setCurrentText(arg("--color"))
    if arg("--screenshot"):
        w.setAttribute(Qt.WA_DontShowOnScreen)
        w.show()
        target = int(arg("--frames", "60"))
        w.set_realtime.emit(False)
        poll = QtCore.QTimer(w)

        def check():
            if w.view.snap is not None and w.view.snap["frame"] >= target:
                poll.stop()
                w.toggle_run()
                QtCore.QTimer.singleShot(400, lambda: (w.screenshot(arg("--screenshot")), w.close(), app.quit()))
        poll.timeout.connect(check)
        poll.start(20)
        QtCore.QTimer.singleShot(800, w.toggle_run)
    else:
        w.show()
    sys.exit(app.exec_())
