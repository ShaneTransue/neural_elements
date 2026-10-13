"""Animated comparisons for the README: FEM, Neural Elements and a mass-spring system, side by side.

    python examples/make_gifs.py [names]   -> examples/images/<name>.gif (all five without names)

Each animation runs the same scene with three force models and shows, frame by frame:
    top row      the three models, colored by von Mises stress (one color scale for all three)
    bottom row   the main measurement of the scene over time (end force, torque or tip deflection), and the
                 position difference of the second and third model from the first, in % of the first model's
                 largest displacement (0 - 10%, the same scale as the viewer)

    compression   linear material, one end pulled out by 32% and pushed in by 32% (the original application's
                  stretch scene): the mass-spring system collapses in compression, FEM and Neural Elements do not
    twist         linear material, one end turned by up to half a turn each way
    cantilever    linear material, bending under its own weight
    neohookean    Neo-Hookean material, out and in by 32%: Neural Elements follow Neo-Hookean FEM, which softens in
                  tension and stiffens in compression, while linear FEM does neither
    mass_spring_nonlinear
                  Neo-Hookean rubber (nu = 0.45), out and in by 32%, against a mass-spring system: the springs are
                  too stiff in tension, too soft in compression and lose a quarter of the volume; Neural Elements
                  follow Neo-Hookean FEM. The plot shows end force against the end's displacement (the force curve).

    materials     linear, Neo-Hookean and St. Venant-Kirchhoff, out and in by 32%, each with Neural Elements (nonlinear
                  formulation, the material's network) and its FEM: one force curve per material, solid FEM and
                  dashed Neural Elements. Neo-Hookean softens in tension and stiffens in compression; St.
                  Venant-Kirchhoff does the opposite.

Rendering uses the viewer's OpenGL renderer on an offscreen surface; Pillow writes the GIFs.
A few minutes on a laptop CPU (the simulations run on the CPU in double precision).
"""
import dataclasses
import math
import os
import sys

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # the library and the viewer live one folder up

from gui.renderer import Camera, Renderer, colormap_table   # noqa: E402
from neural_elements import SCENES, Material, Simulation      # noqa: E402
from neural_elements.scenes import boundary_faces             # noqa: E402

OUT = os.path.join(HERE, "images")
PANEL_W, PANEL_H, TITLE_H, HEADER_H, BAR_W = 300, 190, 34, 34, 70      # titles have room for two lines
DIFF_RANGE = 10.0                                            # % of the first model's largest displacement
COLORS = [QtGui.QColor(25, 25, 25), QtGui.QColor(27, 67, 251), QtGui.QColor(224, 123, 0)]

CLIPS = {
    "compression": dict(
        scene=dataclasses.replace(SCENES["stretch"], drive="cycle"), frames=320, stride=2,
        title="Pulled out by 32%, then pushed in by 32% (linear material)",
        runs=[("FEM, linear corotational", "fem", "linear"), ("Neural Elements", "ne_nonlinear", "linear"),
              ("Mass-spring", "mass_spring", "linear")],
        series="force", ylabel="end force", view=(0.55, 0.38), zoom=0.45),
    "twist": dict(
        scene=SCENES["twist"], frames=320, stride=3,
        title="One end turned by up to half a turn each way (linear material)",
        runs=[("FEM, linear corotational", "fem", "linear"), ("Neural Elements", "ne_nonlinear", "linear"),
              ("Mass-spring", "mass_spring", "linear")],
        series="torque", ylabel="end torque", view=(0.55, 0.38), zoom=0.5),
    "cantilever": dict(
        scene=SCENES["cantilever"], frames=240, stride=2,
        title="Cantilever bending under its own weight (linear material)",
        runs=[("FEM, linear corotational", "fem", "linear"), ("Neural Elements", "ne_nonlinear", "linear"),
              ("Mass-spring", "mass_spring", "linear")],
        series="tip", ylabel="tip deflection", view=(0.0, 0.12), zoom=0.62),
    "neohookean": dict(
        scene=dataclasses.replace(SCENES["stretch"], drive="cycle"), frames=320, stride=2,
        title="Neo-Hookean material, pulled out and pushed in by 32%",
        runs=[("FEM, Neo-Hookean", "fem", "neohookean"), ("Neural Elements, Neo-Hookean", "ne_nonlinear", "neohookean"),
              ("FEM, linear", "fem", "linear")],
        series="force", ylabel="end force", view=(0.55, 0.38), zoom=0.45),
    "mass_spring_nonlinear": dict(
        scene=dataclasses.replace(SCENES["stretch"], drive="cycle"), frames=320, stride=2, nu=0.45,
        title="Neo-Hookean rubber (nu = 0.45), pulled out and pushed in by 32%",
        runs=[("FEM, Neo-Hookean", "fem", "neohookean"), ("Neural Elements, Neo-Hookean", "ne_nonlinear", "neohookean"),
              ("Mass-spring", "mass_spring", "linear")],
        series="force", ylabel="end force", xaxis="strain", volume=True, view=(0.55, 0.38), zoom=0.45),
    "materials": dict(
        scene=dataclasses.replace(SCENES["stretch"], drive="cycle"), frames=320, stride=2, layout="materials",
        title="One formulation, three materials: pulled out and pushed in by 32%",
        runs=[(f"{kind}, {name}", kind_model, material) for kind, kind_model in
              (("Neural Elements", "ne_nonlinear"), ("FEM", "fem"))
              for name, material in (("linear", "linear"), ("Neo-Hookean", "neohookean"),
                                     ("St. Venant-Kirchhoff", "stvk"))],
        series="force", ylabel="end force", view=(0.55, 0.38), zoom=0.45),
}
MATERIAL_COLORS = [QtGui.QColor(70, 70, 70), QtGui.QColor(27, 67, 251), QtGui.QColor(214, 90, 30)]


# ---------------------------------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------------------------------
def simulate(clip):
    """Run every model of the clip; keep positions, von Mises stress and the measurement of every frame."""
    data = []
    for label, model, material in clip["runs"]:
        sim = Simulation(clip["scene"], model, Material(material, nu=clip.get("nu")))
        x, vm = [], []
        for _ in range(clip["frames"]):
            sim.step()
            x.append(sim.positions())
            vm.append(sim.von_mises())
        h = sim.history
        length = float(sim.rest[:, 0].max() - sim.rest[:, 0].min())
        data.append(dict(label=label, x=np.array(x), vm=np.array(vm), series=np.array(h[clip["series"]]),
                         time=np.array(h["time"]), axial=np.array(h["axial"]), drive=np.array(h["drive"]),
                         strain=100 * np.array(h["drive"]) / length, volume=np.array(h["volume"])))
        bad = len(h["converged"]) - sum(h["converged"])
        print(f"    {label:32s} {clip['frames']} frames, unconverged steps {bad}, inverted elements up to "
              f"{max(h['inverted'])}", flush=True)
    return sim.rest.numpy(), sim.elems, sim.boundary.fixed.numpy(), data


# ---------------------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------------------
class Offscreen:
    """An OpenGL 4.1 context on an offscreen surface, a multisampled framebuffer and the viewer's renderer."""

    def __init__(self, w, h):
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        self.ctx = QtGui.QOpenGLContext()
        self.ctx.setFormat(fmt)
        self.ctx.create()
        self.surface = QtGui.QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        self.ctx.makeCurrent(self.surface)
        vp = QtGui.QOpenGLVersionProfile()
        vp.setVersion(4, 1)
        vp.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        gl = self.ctx.versionFunctions(vp)
        gl.initializeOpenGLFunctions()
        ff = QtGui.QOpenGLFramebufferObjectFormat()
        ff.setAttachment(QtGui.QOpenGLFramebufferObject.CombinedDepthStencil)
        ff.setSamples(8)
        self.fbo = QtGui.QOpenGLFramebufferObject(w, h, ff)
        self.renderer = Renderer()
        self.renderer.initialize(gl)
        self.w, self.h = w, h

    def render(self, camera, x, values, cmap):
        self.fbo.bind()
        self.renderer.update(x, values)
        self.renderer.draw(camera, self.w, self.h, cmap, wire=0.18, point_size=3.0, background=False, ground=False)
        self.fbo.release()
        return self.fbo.toImage()


def ticks(lo, hi, n):
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    return [k * step for k in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]


def draw_plot(qp, rect, curves, now, title, xlabel):
    """Curves (dicts: x, y, color, width, dash) up to frame `now`, on fixed axes over the whole run, with a cursor
    at the current x and a dot at each curve's current point."""
    left, bottom = 54, 26
    r = QtCore.QRectF(rect.x() + left, rect.y() + 8, rect.width() - left - 8, rect.height() - bottom - 12)
    xs, ys = np.concatenate([c["x"] for c in curves]), np.concatenate([c["y"] for c in curves])
    xlo, xhi = min(xs.min(), 0.0), max(xs.max(), 0.0)
    xpad = 0.0 if xlo == 0 else 0.06 * (xhi - xlo)          # time starts at the left edge
    xlo, xhi = xlo - xpad, xhi + xpad
    lo, hi = min(ys.min(), 0.0), max(ys.max(), 0.0)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    X = lambda v: r.left() + r.width() * (v - xlo) / (xhi - xlo)
    Y = lambda v: r.top() + r.height() * (1 - (v - lo) / (hi - lo))
    grid, label = QtGui.QPen(QtGui.QColor(232, 234, 238), 1), QtGui.QColor(110, 110, 110)
    qp.setFont(QtGui.QFont("Segoe UI", 7))
    for v in ticks(lo, hi, 4):
        qp.setPen(grid)
        qp.drawLine(QtCore.QPointF(r.left(), Y(v)), QtCore.QPointF(r.right(), Y(v)))
        qp.setPen(label)
        qp.drawText(QtCore.QRectF(rect.x(), Y(v) - 7, left - 4, 14), Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
    room = r.right() - qp.fontMetrics().horizontalAdvance(xlabel) - 24    # tick labels end before the axis name
    for v in ticks(xlo, xhi, max(4, int(r.width() // 60))):
        qp.setPen(grid)
        qp.drawLine(QtCore.QPointF(X(v), r.top()), QtCore.QPointF(X(v), r.bottom()))
        if X(v) < room:
            qp.setPen(label)
            qp.drawText(QtCore.QRectF(X(v) - 20, r.bottom() + 1, 40, 16), Qt.AlignHCenter, f"{v:g}")
    qp.setPen(label)
    qp.drawText(QtCore.QRectF(r.right() - 90, r.bottom() + 1, 90, 16), Qt.AlignRight, xlabel)
    qp.setPen(QtGui.QPen(QtGui.QColor(170, 172, 178), 1))
    qp.drawRect(r)
    qp.drawLine(QtCore.QPointF(r.left(), Y(0)), QtCore.QPointF(r.right(), Y(0)))
    qp.setPen(QtGui.QPen(QtGui.QColor(200, 60, 60), 1, Qt.DashLine))
    qp.drawLine(QtCore.QPointF(X(curves[0]["x"][now]), r.top()), QtCore.QPointF(X(curves[0]["x"][now]), r.bottom()))
    for c in curves:
        pen = QtGui.QPen(c["color"], c["width"])
        if c.get("dash"):
            pen.setStyle(Qt.DashLine)
        qp.setPen(pen)
        qp.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(X(a), Y(b)) for a, b in zip(c["x"][:now + 1], c["y"][:now + 1])]))
        qp.setPen(Qt.NoPen)
        qp.setBrush(c["color"])
        qp.drawEllipse(QtCore.QPointF(X(c["x"][now]), Y(c["y"][now])), 2.5, 2.5)
        qp.setBrush(Qt.NoBrush)
    qp.setFont(QtGui.QFont("Segoe UI", 8))
    qp.setPen(QtGui.QColor(20, 20, 20))
    qp.drawText(QtCore.QRectF(rect.x(), rect.y() - TITLE_H, rect.width(), TITLE_H), Qt.AlignCenter, title)


def model_curves(data, xaxis):
    """The three models' curves: the reference solid and thick, the second dashed on top, the third underneath."""
    style = {0: dict(width=2.6), 1: dict(width=1.8, dash=True), 2: dict(width=1.8)}
    return [dict(x=data[k][xaxis], y=data[k]["series"], color=COLORS[k], **style[k]) for k in (2, 0, 1)]


def draw_colorbar(qp, x, y, h, table, top, bottom, caption):
    for i in range(h):
        c = table[int(255 * (1 - i / (h - 1)))]
        qp.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
        qp.drawLine(x, y + i, x + 10, y + i)
    qp.setPen(QtGui.QColor(60, 60, 60))
    qp.drawRect(x, y, 10, h)
    qp.setFont(QtGui.QFont("Segoe UI", 7))
    qp.drawText(x + 14, y + 8, top)
    qp.drawText(x + 14, y + h, bottom)
    qp.save()
    qp.translate(x + 50, y + h / 2)
    qp.rotate(-90)
    qp.drawText(QtCore.QRectF(-h / 2 - 20, -10, h + 40, 14), Qt.AlignCenter, caption)
    qp.restore()


def render_clip(name, clip, rest, elems, fixed, data, gl):
    faces = boundary_faces(elems)
    gl.renderer.set_mesh(faces, fixed, rest)
    cam = Camera()
    cam.yaw, cam.pitch = clip["view"]
    allx = np.concatenate([d["x"][::10].reshape(-1, 3) for d in data])
    cam.frame(allx)
    cam.dist *= clip["zoom"]
    vmax = float(np.percentile(data[0]["vm"], 99.5))
    scale = float(np.linalg.norm(data[0]["x"] - rest, axis=2).max())    # the first model's largest displacement
    viridis, magma = colormap_table("viridis"), colormap_table("magma_r")
    W, H = 3 * PANEL_W + BAR_W, HEADER_H + 2 * (TITLE_H + PANEL_H) + 6
    frames = []
    for i in range(0, clip["frames"], clip["stride"]):
        img = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(255, 255, 255))
        qp = QtGui.QPainter(img)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setRenderHint(QtGui.QPainter.TextAntialiasing)
        top = HEADER_H + TITLE_H
        for k, d in enumerate(data):                          # top row: von Mises
            qp.drawImage(k * PANEL_W, top, gl.render(cam, d["x"][i], d["vm"][i] / vmax, "viridis"))
        bottom = top + PANEL_H + TITLE_H
        for k in (1, 2):                                      # bottom row: difference from the first model
            diff = 100 * np.linalg.norm(data[k]["x"][i] - data[0]["x"][i], axis=1) / scale
            qp.drawImage(k * PANEL_W, bottom, gl.render(cam, data[k]["x"][i], diff / DIFF_RANGE, "magma_r"))
        xaxis = clip.get("xaxis", "time")
        draw_plot(qp, QtCore.QRectF(0, bottom, PANEL_W, PANEL_H), model_curves(data, xaxis), i,
                  f"{clip['ylabel']} (colors as the titles)", "t [s]" if xaxis == "time" else "stretch [%]")
        # titles and header
        qp.setFont(QtGui.QFont("Segoe UI", 8))
        wrap = Qt.AlignCenter | Qt.TextWordWrap                # long titles go on two lines, inside their panel
        for k, d in enumerate(data):
            qp.setPen(COLORS[k])
            title = d["label"] + (f"\nvolume {100 * (d['volume'][i] - 1):+.1f}%" if clip.get("volume") else "")
            qp.drawText(QtCore.QRectF(k * PANEL_W + 6, HEADER_H, PANEL_W - 12, TITLE_H), wrap, title)
            if k:
                qp.drawText(QtCore.QRectF(k * PANEL_W + 6, top + PANEL_H, PANEL_W - 12, TITLE_H), Qt.AlignCenter,
                            f"{d['label']}\ndifference from {data[0]['label']}")
        qp.setPen(QtGui.QColor(20, 20, 20))
        qp.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.DemiBold))
        qp.drawText(QtCore.QRectF(10, 0, W - 20, HEADER_H), Qt.AlignLeft | Qt.AlignVCenter, clip["title"])
        qp.setFont(QtGui.QFont("Segoe UI", 9))
        d0 = data[0]
        state = (f"end rotation {math.degrees(d0['drive'][i]):+4.0f} deg" if name == "twist" else
                 f"tip {d0['series'][i]:+.2f}" if name == "cantilever" else f"stretch {d0['strain'][i]:+3.0f}%")
        qp.drawText(QtCore.QRectF(10, 0, W - 20, HEADER_H), Qt.AlignRight | Qt.AlignVCenter,
                    f"{state}     t = {d0['time'][i]:.2f} s")
        draw_colorbar(qp, 3 * PANEL_W + 6, top + 14, PANEL_H - 28, viridis, f"{vmax:.0f}", "0", "von Mises stress")
        draw_colorbar(qp, 3 * PANEL_W + 6, bottom + 14, PANEL_H - 28, magma, f"{DIFF_RANGE:g}%", "0%",
                      "difference, % of displacement")
        qp.end()
        frames.append(to_pil(img))
    return frames


def render_materials(clip, rest, elems, fixed, data, gl):
    """Top: Neural Elements for each material (von Mises, volume and difference from that material's FEM in the
    titles); bottom: one wide plot of end force against stretch, solid FEM and dashed Neural Elements."""
    ne, fem = data[:3], data[3:]
    gl.renderer.set_mesh(boundary_faces(elems), fixed, rest)
    cam = Camera()
    cam.yaw, cam.pitch = clip["view"]
    cam.frame(np.concatenate([d["x"][::10].reshape(-1, 3) for d in data]))
    cam.dist *= clip["zoom"]
    vmax = float(np.percentile(np.concatenate([d["vm"] for d in fem]), 99.5))
    scales = [float(np.linalg.norm(d["x"] - rest, axis=2).max()) for d in fem]
    viridis = colormap_table("viridis")
    W, H = 3 * PANEL_W + BAR_W, HEADER_H + 2 * (TITLE_H + PANEL_H) + 6
    curves = [dict(x=d["strain"], y=d["series"], color=MATERIAL_COLORS[k], width=3.2) for k, d in enumerate(fem)]
    curves += [dict(x=d["strain"], y=d["series"], color=MATERIAL_COLORS[k].darker(150), width=1.4, dash=True)
               for k, d in enumerate(ne)]
    frames = []
    for i in range(0, clip["frames"], clip["stride"]):
        img = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(255, 255, 255))
        qp = QtGui.QPainter(img)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        top = HEADER_H + TITLE_H
        qp.setFont(QtGui.QFont("Segoe UI", 8))
        for k, d in enumerate(ne):
            qp.drawImage(k * PANEL_W, top, gl.render(cam, d["x"][i], d["vm"][i] / vmax, "viridis"))
            diff = 100 * float(np.sqrt((np.linalg.norm(d["x"][i] - fem[k]["x"][i], axis=1) ** 2).mean())) / scales[k]
            qp.setPen(MATERIAL_COLORS[k])
            qp.drawText(QtCore.QRectF(k * PANEL_W + 6, HEADER_H, PANEL_W - 12, TITLE_H), Qt.AlignCenter,
                        f"{d['label']}\nvolume {100 * (d['volume'][i] - 1):+.1f}%,  {diff:.2f}% from FEM")
        bottom = top + PANEL_H + TITLE_H
        draw_plot(qp, QtCore.QRectF(0, bottom, 3 * PANEL_W, PANEL_H), curves, i,
                  "end force against stretch: thick solid FEM, dashed Neural Elements (colors as the titles)",
                  "stretch [%]")
        qp.setPen(QtGui.QColor(20, 20, 20))
        qp.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.DemiBold))
        qp.drawText(QtCore.QRectF(10, 0, W - 20, HEADER_H), Qt.AlignLeft | Qt.AlignVCenter, clip["title"])
        qp.setFont(QtGui.QFont("Segoe UI", 9))
        qp.drawText(QtCore.QRectF(10, 0, W - 20, HEADER_H), Qt.AlignRight | Qt.AlignVCenter,
                    f"stretch {fem[0]['strain'][i]:+3.0f}%     t = {fem[0]['time'][i]:.2f} s")
        draw_colorbar(qp, 3 * PANEL_W + 6, top + 14, PANEL_H - 28, viridis, f"{vmax:.0f}", "0", "von Mises stress")
        qp.end()
        frames.append(to_pil(img))
    return frames


def to_pil(img):
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    a = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[:, :3 * img.width()]
    return Image.fromarray(a.reshape(img.height(), img.width(), 3).copy())


def save_gif(frames, path, ms=50):
    pal = [f.quantize(colors=192, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, disposal=1)


def main():
    names = sys.argv[1:] or list(CLIPS)
    app = QtGui.QGuiApplication(sys.argv[:1])               # noqa: F841  (needed for the OpenGL context)
    gl = Offscreen(PANEL_W, PANEL_H)
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        clip = CLIPS[name]
        print(f"{name}: simulating", flush=True)
        rest, elems, fixed, data = simulate(clip)
        if clip.get("layout") == "materials":
            frames = render_materials(clip, rest, elems, fixed, data, gl)
        else:
            frames = render_clip(name, clip, rest, elems, fixed, data, gl)
        path = os.path.join(OUT, f"{name}.gif")
        save_gif(frames, path)
        print(f"  {len(frames)} frames -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
