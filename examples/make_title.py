"""The title image of the README: one element taken out of a deformed body, with its reaction forces.

    python examples/make_title.py            -> examples/images/title.png

Background: a twisted Neo-Hookean beam (finite elements of the material), rendered with the viewer's OpenGL renderer
and faded. Foreground: one of its surface elements, drawn as the reference element under that element's deformation
(exaggerated for visibility), with its undeformed outline dashed and its reaction forces f_i = V P(F) grad N_i as
arrows: the quantity the Neural Element networks are trained on.
"""
import math
import os
import sys

import numpy as np
import torch
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # the library and the viewer live one folder up
sys.path.insert(0, HERE)

from make_gifs import Offscreen                              # noqa: E402
from gui.renderer import Camera                              # noqa: E402
from neural_elements import Material, Simulation              # noqa: E402
from neural_elements.geometry import reference_tetrahedron    # noqa: E402
from neural_elements.materials import neohookean_stress       # noqa: E402
from neural_elements.nonlinear import REFERENCE_G, REFERENCE_V  # noqa: E402
from neural_elements.scenes import boundary_faces             # noqa: E402

W, H = 1600, 520
INK, ACCENT, FORCE = QtGui.QColor(24, 34, 64), QtGui.QColor(27, 67, 251), QtGui.QColor(232, 110, 20)


def background():
    """The twisted beam at about 150 degrees of end rotation, as an image, its camera and the simulation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = Simulation("twist", "fem", Material("neohookean"), refine=2, device=device)
    sim.run(50)
    x = sim.positions()
    gl = Offscreen(W, H)
    faces = boundary_faces(sim.elems)
    gl.renderer.set_mesh(faces, np.zeros(sim.N, bool), sim.rest.numpy())
    cam = Camera()
    cam.yaw, cam.pitch = math.radians(28), math.radians(24)
    cam.frame(x)
    cam.dist *= 0.42
    cam.target = cam.target + np.array([-1.2, -0.2, 0.0])       # the body across the image, faded under the title
    vm = sim.von_mises()
    gl.fbo.bind()
    gl.renderer.update(x, vm / np.percentile(vm, 99))
    gl.renderer.draw(cam, W, H, "viridis", wire=0.35, show_fixed=False, background=False, ground=False)
    gl.fbo.release()
    return gl.fbo.toImage(), cam, sim, x, faces


def projector(cam):
    """World point -> pixel, with the camera's own matrices."""
    view, proj = cam.matrices(W / H)
    M = np.array(proj.data()).reshape(4, 4).T @ np.array(view.data()).reshape(4, 4).T

    def to_pixel(p):
        q = M @ np.append(p, 1.0)
        return np.array([(q[0] / q[3] + 1) * W / 2, (1 - q[1] / q[3]) * H / 2]), q[3]
    return to_pixel


def pick_element(sim, x, faces, to_pixel, target):
    """The surface face nearest the camera among those close to a target pixel, and its element."""
    best, best_depth = None, 1e30
    eye, _ = Camera.basis(cam_holder[0])
    for f in faces:
        c = x[f].mean(0)
        n = np.cross(x[f[1]] - x[f[0]], x[f[2]] - x[f[0]])
        if np.dot(n, eye - c) <= 0:
            continue
        pix, depth = to_pixel(c)
        if np.linalg.norm(pix - target) < 40 and depth < best_depth:
            best, best_depth = f, depth
    elems = sim.elems.numpy()
    e = next(k for k, t in enumerate(elems) if set(best) <= set(t))
    return best, e


cam_holder = []


def arrow(qp, a, b, color, width=3.0, head=12.0):
    qp.setPen(QtGui.QPen(color, width, Qt.SolidLine, Qt.RoundCap))
    qp.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    tip = [b + 2 * d, b - head * d + 0.45 * head * n, b - head * d - 0.45 * head * n]
    qp.setPen(Qt.NoPen)
    qp.setBrush(color)
    qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in tip]))
    qp.setBrush(Qt.NoBrush)


def main():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv[:1])               # noqa: F841
    img, cam, sim, x, faces = background()
    cam_holder.append(cam)
    to_pixel = projector(cam)
    face, e = pick_element(sim, x, faces, to_pixel, np.array([0.60 * W, 0.62 * H]))

    # an illustrative deformation of the reference element (a stretch with some shear, larger than in the body so
    # that it reads at this size), and the Neo-Hookean reaction forces it produces
    U = np.array([[1.38, 0.10, 0.0], [0.10, 0.84, 0.0], [0.0, 0.0, 0.88]])
    Xr = reference_tetrahedron().numpy()
    Xr = Xr - Xr.mean(0)
    xr = Xr @ U.T
    P = neohookean_stress(torch.tensor(U)[None], torch.tensor([0.3]))[0].numpy()
    f = REFERENCE_V.item() * (P[None] @ REFERENCE_G.numpy()[:, :, None])[:, :, 0]     # f_i = V P grad N_i

    # an oblique view of the enlarged element
    yaw, pitch = math.radians(-38), math.radians(22)
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R = Rx @ Ry
    center, scale = np.array([0.795 * W, 0.50 * H]), 150.0
    flat = lambda p: center + scale * np.array([(R @ p)[0], -(R @ p)[1]])
    depth = lambda p: (R @ p)[2]

    qp = QtGui.QPainter(img)
    qp.setRenderHint(QtGui.QPainter.Antialiasing)
    # fade the body: strongly on the left (title), lightly on the right
    g = QtGui.QLinearGradient(0, 0, W, 0)
    g.setColorAt(0.0, QtGui.QColor(255, 255, 255, 250))
    g.setColorAt(0.36, QtGui.QColor(255, 255, 255, 225))
    g.setColorAt(0.62, QtGui.QColor(255, 255, 255, 120))
    g.setColorAt(1.0, QtGui.QColor(255, 255, 255, 95))
    qp.fillRect(0, 0, W, H, g)

    # the chosen element on the body, and the callout to its enlarged copy
    pts = [to_pixel(x[v])[0] for v in face]
    qp.setPen(QtGui.QPen(ACCENT, 2.5))
    qp.setBrush(QtGui.QColor(27, 67, 251, 90))
    qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in pts]))
    qp.setBrush(Qt.NoBrush)
    c0 = np.mean(pts, axis=0)
    qp.setPen(QtGui.QPen(QtGui.QColor(27, 67, 251, 160), 1.6, Qt.DashLine))
    for v in range(4):
        qp.drawLine(QtCore.QPointF(*c0), QtCore.QPointF(*flat(xr[v])))

    # a soft plate behind the enlarged element
    qp.setPen(Qt.NoPen)
    halo = QtGui.QRadialGradient(QtCore.QPointF(*center), 300)
    halo.setColorAt(0.0, QtGui.QColor(255, 255, 255, 245))
    halo.setColorAt(0.6, QtGui.QColor(255, 255, 255, 225))
    halo.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
    qp.setBrush(halo)
    qp.drawEllipse(QtCore.QPointF(*center), 300, 300)
    qp.setBrush(Qt.NoBrush)

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    qp.setPen(QtGui.QPen(QtGui.QColor(120, 125, 140), 1.6, Qt.DashLine))     # the undeformed reference
    for a, b in edges:
        qp.drawLine(QtCore.QPointF(*flat(Xr[a])), QtCore.QPointF(*flat(Xr[b])))
    tris = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]                        # the deformed element, back to front
    light = np.array([0.3, 0.6, 1.0]) / np.linalg.norm([0.3, 0.6, 1.0])
    for t in sorted(tris, key=lambda t: -np.mean([depth(xr[v]) for v in t])):
        p = [R @ xr[v] for v in t]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        shade = abs(np.dot(n / np.linalg.norm(n), light))
        col = QtGui.QColor(int(60 + 120 * shade), int(100 + 110 * shade), 245, 120)
        qp.setPen(QtGui.QPen(INK, 2.2))
        qp.setBrush(col)
        qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*flat(xr[v])) for v in t]))
    qp.setBrush(Qt.NoBrush)
    fmax = np.linalg.norm(f, axis=1).max()
    tips = []
    for v in range(4):                                                          # reaction forces
        a = flat(xr[v])
        d = np.array([(R @ f[v])[0], -(R @ f[v])[1]])
        tips.append(a + 125 * d / fmax)
        arrow(qp, a, tips[-1], FORCE, width=3.4, head=14)
    for v in range(4):                                                          # nodes
        qp.setPen(QtGui.QPen(INK, 2.2))
        qp.setBrush(QtGui.QColor(255, 255, 255))
        qp.drawEllipse(QtCore.QPointF(*flat(xr[v])), 7, 7)
    qp.setBrush(Qt.NoBrush)
    small = QtGui.QFont("Segoe UI", 12)
    small.setItalic(True)
    qp.setFont(small)
    box = np.array([flat(p) for p in np.concatenate([Xr, xr])] + tips)        # labels outside the drawing
    qp.setPen(FORCE)
    qp.drawText(QtCore.QRectF(box[:, 0].max() - 200, box[:, 1].min() - 40, 220, 26), Qt.AlignRight | Qt.TextDontClip,
                "reaction forces")
    qp.setPen(QtGui.QColor(95, 100, 118))
    qp.drawText(QtCore.QRectF(center[0] - 350, box[:, 1].max() + 10, 700, 26), Qt.AlignHCenter | Qt.TextDontClip,
                "one element under its deformation   (dashed: at rest)")

    # title
    qp.setPen(INK)
    qp.setFont(QtGui.QFont("Segoe UI", 46, QtGui.QFont.Bold))
    qp.drawText(QtCore.QPointF(70, 205), "Neural Elements")
    qp.setFont(QtGui.QFont("Segoe UI", 17))
    qp.setPen(QtGui.QColor(55, 62, 85))
    qp.drawText(QtCore.QPointF(74, 262), "Finite elements whose internal forces are learned")
    qp.drawText(QtCore.QPointF(74, 294), "from the reaction forces of one reference element")
    qp.setFont(QtGui.QFont("Segoe UI", 13))
    qp.setPen(QtGui.QColor(100, 106, 125))
    qp.drawText(QtCore.QPointF(75, 350), "Data-driven Linear and Non-linear Elastic Materials   ·   PyTorch (CPU/GPU)")
    qp.end()
    out = os.path.join(HERE, "images", "title.png")
    img.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
