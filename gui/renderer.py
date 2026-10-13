"""OpenGL drawing of a simulation with GLSL shaders (OpenGL 4.1 core profile, through PyQt5's QtGui OpenGL classes).

What is drawn, back to front:
    background   a vertical gradient (one full-screen triangle)
    ground       a large quad whose grid lines are computed per pixel in the fragment shader
    surface      the boundary triangles of the mesh: flat shading from the screen-space derivatives of the position
                 (no normals to upload), a color map looked up in a 1D texture, and an optional wireframe drawn in
                 the same pass from barycentric coordinates
    ghost        the FEM reference: the same mesh drawn as translucent edges only
    points       the constrained nodes, as round point sprites

The renderer only needs a current OpenGL context, so it serves both the interactive window and offscreen rendering
(screenshots, animation frames). Per frame it uploads the corner positions and color values of the surface triangles.
"""
import math

import numpy as np
from matplotlib import colormaps
from PyQt5 import QtGui
from PyQt5.QtGui import QMatrix4x4, QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, \
    QOpenGLVertexArrayObject, QVector3D

GL_FLOAT, GL_TRIANGLES, GL_POINTS = 0x1406, 0x0004, 0x0000
GL_DEPTH_TEST, GL_BLEND, GL_PROGRAM_POINT_SIZE, GL_MULTISAMPLE = 0x0B71, 0x0BE2, 0x8642, 0x809D
GL_ONE, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 1, 0x0302, 0x0303
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100

BACKGROUND_VS = """#version 410 core
out float v_y;
void main() {                                   // one triangle covering the screen
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    v_y = p.y * 0.5 + 0.5;
    gl_Position = vec4(p, 0.999, 1.0);
}"""
BACKGROUND_FS = """#version 410 core
in float v_y;
out vec4 color;
void main() { color = vec4(mix(vec3(0.86, 0.88, 0.91), vec3(0.985, 0.985, 0.99), v_y), 1.0); }"""

GROUND_VS = """#version 410 core
layout(location = 0) in vec3 position;
uniform mat4 view, projection;
out vec3 v_world;
void main() { v_world = position; gl_Position = projection * view * vec4(position, 1.0); }"""
GROUND_FS = """#version 410 core
in vec3 v_world;
uniform vec2 center;
uniform float extent;
out vec4 color;
float grid(vec2 p) {                            // 1 on a grid line, about one pixel wide at any distance
    vec2 w = fwidth(p);
    vec2 d = abs(fract(p - 0.5) - 0.5) / w;
    return 1.0 - min(min(d.x, d.y), 1.0);
}
void main() {
    float fade = 1.0 - smoothstep(0.35 * extent, extent, length(v_world.xz - center));
    float a = 0.07 * grid(v_world.xz) + 0.20 * grid(v_world.xz / 5.0);
    color = vec4(vec3(0.45, 0.47, 0.50), a * fade);
}"""

SURFACE_VS = """#version 410 core
layout(location = 0) in vec3 position;
layout(location = 1) in float value;             // color value, already scaled to 0..1
layout(location = 2) in vec3 bary;               // barycentric corner of the triangle (for the wireframe)
uniform mat4 view, projection;
out vec3 v_eye;
out float v_value;
out vec3 v_bary;
void main() {
    vec4 p = view * vec4(position, 1.0);
    v_eye = p.xyz;
    v_value = value;
    v_bary = bary;
    gl_Position = projection * p;
}"""
SURFACE_FS = """#version 410 core
in vec3 v_eye;
in float v_value;
in vec3 v_bary;
uniform sampler2D colormap;
uniform bool use_colormap;
uniform bool ghost;                              // edges only, translucent (the FEM reference)
uniform float wire;                              // strength of the wireframe on the surface
uniform vec3 base_color;
out vec4 color;
float edge() {                                   // 1 on a triangle edge, 0 inside
    vec3 a = smoothstep(vec3(0.0), 1.3 * fwidth(v_bary), v_bary);
    return 1.0 - min(min(a.x, a.y), a.z);
}
void main() {
    float e = edge();
    if (ghost) {
        if (e < 0.02) discard;
        color = vec4(0.05, 0.05, 0.08, 0.45 * e);
        return;
    }
    vec3 n = normalize(cross(dFdx(v_eye), dFdy(v_eye)));    // flat normal from screen-space derivatives
    vec3 v = normalize(-v_eye);
    if (dot(n, v) < 0.0) n = -n;
    vec3 h = normalize(v + normalize(vec3(0.35, 0.85, 0.4)));
    vec3 albedo = use_colormap ? texture(colormap, vec2(clamp(v_value, 0.0, 1.0), 0.5)).rgb : base_color;
    vec3 c = albedo * (0.30 + 0.70 * dot(n, v)) + 0.10 * pow(max(dot(n, h), 0.0), 48.0);
    color = vec4(mix(c, vec3(0.10), wire * 0.6 * e), 1.0);
}"""

POINTS_VS = """#version 410 core
layout(location = 0) in vec3 position;
uniform mat4 view, projection;
uniform float size;
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    gl_Position.z -= 0.002 * gl_Position.w;     // drawn just in front of the surface they lie on
    gl_PointSize = size;
}"""
POINTS_FS = """#version 410 core
out vec4 color;
void main() {
    vec2 c = 2.0 * gl_PointCoord - 1.0;
    if (dot(c, c) > 1.0) discard;               // round points
    color = vec4(0.78, 0.12, 0.12, 1.0);
}"""


def colormap_table(name, n=256):
    """(n, 3) uint8 colors of a matplotlib color map."""
    return (colormaps[name](np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)


class Camera:
    """Orbit camera: yaw and pitch around a target point at a distance."""

    def __init__(self):
        self.yaw, self.pitch, self.dist = math.radians(35), math.radians(20), 20.0
        self.target = np.zeros(3)
        self.fov = 40.0

    def basis(self):
        """Eye position and the camera axes (right, up, back) as rows."""
        cp, sp, cy, sy = math.cos(self.pitch), math.sin(self.pitch), math.cos(self.yaw), math.sin(self.yaw)
        eye = self.target + self.dist * np.array([cp * sy, sp, cp * cy])
        f = (self.target - eye) / np.linalg.norm(self.target - eye)
        r = np.cross(f, [0, 1, 0])
        r /= np.linalg.norm(r)
        return eye, np.stack([r, np.cross(r, f), -f])

    def matrices(self, aspect):
        eye, _ = self.basis()
        view, proj = QMatrix4x4(), QMatrix4x4()
        view.lookAt(QVector3D(*eye), QVector3D(*self.target), QVector3D(0, 1, 0))
        proj.perspective(self.fov, aspect, max(0.05, self.dist * 0.01), self.dist * 20 + 100)
        return view, proj

    def frame(self, x):
        """Look at the bounding box of the points x (N, 3)."""
        lo, hi = x.min(0), x.max(0)
        self.target = (lo + hi) / 2
        self.dist = 0.8 * float(np.linalg.norm(hi - lo)) / math.tan(math.radians(self.fov) / 2) + 2


class Renderer:
    def __init__(self):
        self.faces = None
        self.ready = False

    # setup -----------------------------------------------------------------------------------------------------------
    def initialize(self, gl):
        """Compile the shaders and create the buffers (needs a current OpenGL 4.1 context)."""
        self.gl = gl
        self.background = self._program(BACKGROUND_VS, BACKGROUND_FS)
        self.ground = self._program(GROUND_VS, GROUND_FS)
        self.surface = self._program(SURFACE_VS, SURFACE_FS)
        self.points = self._program(POINTS_VS, POINTS_FS)
        self.empty_vao = self._vao()
        self.ground_vao, (self.ground_buf,) = self._vao([3])
        self.mesh_vao, (self.pos_buf, self.val_buf, self.bary_buf) = self._vao([3, 1, 3])
        self.ghost_vao, (self.ghost_buf, self.ghost_bary_buf) = self._vao([3, None, 3])
        self.points_vao, (self.points_buf,) = self._vao([3])
        self.textures = {name: self._texture(colormap_table(name)) for name in ("viridis", "magma_r")}
        self.ready = True

    def _program(self, vs, fs):
        p = QOpenGLShaderProgram()
        ok = p.addShaderFromSourceCode(QOpenGLShader.Vertex, vs) and p.addShaderFromSourceCode(QOpenGLShader.Fragment, fs)
        if not (ok and p.link()):
            raise RuntimeError(f"shader compilation failed:\n{p.log()}")
        return p

    def _vao(self, sizes=()):
        """A vertex array with one buffer per attribute location (size None: location unused)."""
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        bufs = []
        for loc, size in enumerate(sizes):
            if size is None:
                continue
            b = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            b.create()
            b.setUsagePattern(QOpenGLBuffer.DynamicDraw)
            b.bind()
            b.allocate(4)
            self.surface.enableAttributeArray(loc)                      # (any program sets the vertex array state)
            self.surface.setAttributeBuffer(loc, GL_FLOAT, 0, size, 0)
            b.release()
            bufs.append(b)
        vao.release()
        return (vao, bufs) if sizes else vao

    def _texture(self, table):
        img = QtGui.QImage(np.ascontiguousarray(np.stack([table, table])).tobytes(), len(table), 2, 3 * len(table),
                           QtGui.QImage.Format_RGB888).copy()
        t = QOpenGLTexture(img)
        t.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
        t.setWrapMode(QOpenGLTexture.ClampToEdge)
        return t

    @staticmethod
    def _upload(buf, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        buf.bind()
        buf.allocate(data.tobytes(), data.nbytes)
        buf.release()

    # data ------------------------------------------------------------------------------------------------------------
    def set_mesh(self, faces, fixed, x):
        """faces (Nf, 3) boundary triangles; fixed (N,) bool constrained nodes (only those on the surface are drawn);
        x (N, 3) rest positions (the ground is placed below them)."""
        self.faces = faces
        self.ground_y = float(x[:, 1].min()) - 1.5
        bary = np.tile(np.eye(3, dtype=np.float32), (len(faces), 1))
        self._upload(self.bary_buf, bary)
        self._upload(self.ghost_bary_buf, bary)
        on_surface = np.zeros(len(fixed), bool)
        on_surface[faces.ravel()] = True
        self.fixed_nodes = np.where(fixed & on_surface)[0]

    def update(self, x, values=None, xref=None):
        """x (N, 3) node positions; values (N,) color values scaled to 0..1 or None; xref the FEM reference or None."""
        self._upload(self.pos_buf, x[self.faces].reshape(-1, 3))
        self._upload(self.val_buf, (values if values is not None else np.zeros(len(x)))[self.faces].reshape(-1))
        self.has_values = values is not None
        self.has_ghost = xref is not None
        if xref is not None:
            self._upload(self.ghost_buf, xref[self.faces].reshape(-1, 3))
        self._upload(self.points_buf, x[self.fixed_nodes])

    # drawing ---------------------------------------------------------------------------------------------------------
    def draw(self, camera, width, height, cmap="viridis", wire=0.0, show_fixed=True, point_size=7.0,
             background=True, ground=True):
        """wire: strength of the element edges (0 none, 1 dark); background, ground: the gradient and the grid
        (off for a plain white image, e.g. animation frames)."""
        gl = self.gl
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_MULTISAMPLE)
        gl.glDisable(GL_DEPTH_TEST)
        self.background.bind()
        if background:
            self.empty_vao.bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        if self.faces is None:
            self.background.release()
            return
        view, proj = camera.matrices(width / max(height, 1))
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_BLEND)
        # blend colors by alpha, but keep the image opaque (the window composes it with its alpha channel)
        gl.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        # ground: a large quad under the target, faded out towards its border
        ext = max(30.0, camera.dist * 2) if ground else 0.0
        cx, cz, y = camera.target[0], camera.target[2], self.ground_y
        quad = np.array([[cx - ext, y, cz - ext], [cx + ext, y, cz - ext], [cx + ext, y, cz + ext],
                         [cx - ext, y, cz - ext], [cx + ext, y, cz + ext], [cx - ext, y, cz + ext]])
        self._upload(self.ground_buf, quad)
        gl.glDepthMask(False)
        self._use(self.ground, view, proj)
        self.ground.setUniformValue("center", QtGui.QVector2D(cx, cz))
        self.ground.setUniformValue("extent", float(ext))
        self.ground_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 6)
        gl.glDepthMask(True)

        # surface
        n = 3 * len(self.faces)
        self._use(self.surface, view, proj)
        self.textures[cmap].bind(0)
        self.surface.setUniformValue("colormap", 0)
        self.surface.setUniformValue("use_colormap", bool(self.has_values))
        self.surface.setUniformValue("ghost", False)
        self.surface.setUniformValue("wire", float(wire))
        self.surface.setUniformValue("base_color", QVector3D(0.71, 0.78, 0.88))
        self.mesh_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, n)

        # FEM reference: translucent edges, not written to the depth buffer
        if self.has_ghost:
            gl.glDepthMask(False)
            self.surface.setUniformValue("ghost", True)
            self.ghost_vao.bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, n)
            gl.glDepthMask(True)

        if show_fixed and len(self.fixed_nodes):
            gl.glEnable(GL_PROGRAM_POINT_SIZE)
            self._use(self.points, view, proj)
            self.points.setUniformValue("size", float(point_size))
            self.points_vao.bind()
            gl.glDrawArrays(GL_POINTS, 0, len(self.fixed_nodes))
        self.points_vao.release()          # (binds no vertex array)
        self.points.release()              # (uses no program), so that QPainter can draw on top
        gl.glDisable(GL_DEPTH_TEST)
        gl.glDisable(GL_BLEND)

    def _use(self, program, view, proj):
        program.bind()
        program.setUniformValue("view", view)
        program.setUniformValue("projection", proj)
