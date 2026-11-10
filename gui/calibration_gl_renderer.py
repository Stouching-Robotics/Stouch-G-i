"""OpenGL 2.1 hand/depth/wireframe rendering without a visible GL window."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QVector2D
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat, QOpenGLShader, QOpenGLShaderProgram)

from gui.calibration_target_shader import TargetWireframeShader


_VERTEX = '''#version 120
attribute vec3 a_position;
attribute vec3 a_edge;
attribute float a_facing;
attribute float a_shade;
uniform vec2 u_viewport;
uniform vec2 u_zmap;
uniform float u_bias;
varying vec3 v_edge;
varying float v_facing;
varying float v_shade;
varying float v_depth;
void main() {
    gl_Position = vec4(2.0*a_position.x/u_viewport.x-1.0,
        1.0-2.0*a_position.y/u_viewport.y,
        -(a_position.z-u_zmap.x)*u_zmap.y+u_bias, 1.0);
    v_edge=a_edge; v_facing=a_facing; v_shade=a_shade; v_depth=a_position.z;
}
'''
_FRAGMENT = '''#version 120
uniform int u_mode;
uniform vec2 u_target_range;
varying vec3 v_edge;
varying float v_facing;
varying float v_shade;
varying float v_depth;
void main() {
    if (u_mode == 0) {
        float a=225.0/255.0;
        gl_FragColor=vec4(vec3(223.0,185.0,150.0)/255.0*v_shade*a,a);
    } else {
        float front=.25+.75*smoothstep(-.2,.8,v_facing);
        float depth=u_target_range.y-u_target_range.x > 1e-9 ?
            smoothstep(u_target_range.x,u_target_range.y,v_depth) : 1.0;
        float visibility=front*(.2+.8*depth);
        float sigma=.5+.85*(1.0-visibility);
        float d=max(min(min(v_edge.x,v_edge.y),v_edge.z),0.0);
        float wire=exp(-.5*d*d/(sigma*sigma))*(.5/sigma)*(18.0+158.0*visibility)/255.0;
        float fill=(6.0+26.0*visibility)/255.0;
        float a=fill+(1.0-fill)*wire;
        gl_FragColor=vec4(vec3(125.0,188.0,245.0)/255.0*a,a);
    }
}
'''


class CalibrationGLRenderer:
    def __init__(self, *, samples=4):
        app = QGuiApplication.instance()
        if app is None or QThread.currentThread() != app.thread():
            raise RuntimeError('Calibration OpenGL requires the GUI thread')
        self._context = self._surface = self._program = self._buffer = self._fbo = None
        self._samples = samples
        previous = QOpenGLContext.currentContext()
        previous_surface = previous.surface() if previous is not None else None
        try:
            fmt = QSurfaceFormat()
            fmt.setVersion(2, 1)
            fmt.setDepthBufferSize(24)
            self._surface = QOffscreenSurface()
            self._surface.setFormat(fmt)
            self._surface.create()
            self._context = QOpenGLContext()
            self._context.setFormat(self._surface.format())
            if (not self._surface.isValid() or not self._context.create()
                    or not self._context.makeCurrent(self._surface) or self._context.isOpenGLES()):
                raise RuntimeError('A desktop OpenGL context is unavailable')
            self._gl = self._context.functions()
            self._gl.initializeOpenGLFunctions()
            self.info = {}
            for name, token in (('vendor', 0x1f00), ('renderer', 0x1f01), ('version', 0x1f02)):
                value = self._gl.glGetString(token)
                self.info[name] = value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value or '')
            if not self.info['renderer']:
                raise RuntimeError('OpenGL renderer identity is unavailable')
            description = (self.info['vendor'] + ' ' + self.info['renderer']).lower()
            if any(name in description for name in ('llvmpipe', 'softpipe', 'software',
                    'gdi generic', 'swiftshader', 'microsoft basic render driver')):
                raise RuntimeError('OpenGL reports a software rasterizer; using CPU compatibility')
            self._program = QOpenGLShaderProgram()
            for kind, source in ((QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX),
                                 (QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT)):
                if not self._program.addShaderFromSourceCode(kind, source):
                    raise RuntimeError('Calibration shader compilation failed: ' + self._program.log())
            if not self._program.link():
                raise RuntimeError('Calibration shader link failed: ' + self._program.log())
            self._uniforms = {name: self._program.uniformLocation(name) for name in
                ('u_mode', 'u_bias', 'u_viewport', 'u_zmap', 'u_target_range')}
            self._buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not self._buffer.create():
                raise RuntimeError('Calibration vertex buffer creation failed')
        except Exception:
            self.close()
            raise
        finally:
            if self._context is not None:
                self._context.doneCurrent()
            if previous is not None and previous.isValid() and previous_surface is not None:
                previous.makeCurrent(previous_surface)

    def _vertices(self, points, faces, *, target):
        triangles = points[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        shade = .57 + .43 * np.abs(normals @ [.25, -.45, .857]) / np.maximum(
            np.linalg.norm(normals, axis=1), 1e-12)
        attributes = np.zeros((len(faces), 3, 8), dtype=np.float32)
        attributes[:, :, :3] = triangles
        attributes[:, :, 7] = shade[:, None]
        if target:
            facing = TargetWireframeShader(faces)._normal_facing(points)
            attributes[:, :, 6] = facing[faces]
            xy = triangles[:, :, :2]
            twice_area = np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
            for corner in range(3):
                edge = xy[:, (corner + 1) % 3] - xy[:, (corner + 2) % 3]
                attributes[:, corner, 3 + corner] = twice_area / np.maximum(np.linalg.norm(edge, axis=1), 1e-15)
        return attributes.reshape(-1, 8)

    def _upload(self, vertices):
        if not self._buffer.bind():
            raise RuntimeError('Calibration vertex buffer bind failed')
        data = vertices.tobytes()
        self._buffer.allocate(data, len(data))
        for name, offset, size in (('a_position', 0, 3), ('a_edge', 12, 3),
                                   ('a_facing', 24, 1), ('a_shade', 28, 1)):
            location = self._program.attributeLocation(name)
            self._program.enableAttributeArray(location)
            self._program.setAttributeBuffer(location, 0x1406, offset, size, 32)

    def _surface_pass(self, vertices, *, target, bias):
        gl = self._gl
        self._upload(vertices)
        gl.glUniform1i(self._uniforms['u_mode'], int(target))
        gl.glUniform1f(self._uniforms['u_bias'], float(bias))
        gl.glDisable(0x0be2)  # BLEND
        gl.glColorMask(False, False, False, False)
        gl.glDepthMask(True)
        gl.glDepthFunc(0x0201)  # LESS: live wins coplanar target ties via bias.
        gl.glDrawArrays(0x0004, 0, len(vertices))
        gl.glColorMask(True, True, True, True)
        gl.glDepthMask(False)
        gl.glDepthFunc(0x0203)  # LEQUAL: only the nearest prepassed surface.
        gl.glEnable(0x0be2)
        gl.glBlendFunc(1, 0x0303)  # Premultiplied ONE, ONE_MINUS_SRC_ALPHA.
        gl.glDrawArrays(0x0004, 0, len(vertices))

    def render(self, target_px, live_px, faces, width, height):
        if QThread.currentThread() != self._context.thread():
            raise RuntimeError('Calibration OpenGL rendering requires the GUI thread')
        if not self._context.isValid():
            raise RuntimeError('Calibration OpenGL context was lost')
        previous = QOpenGLContext.currentContext()
        previous_surface = previous.surface() if previous is not None else None
        made_current = False
        try:
            made_current = self._context.makeCurrent(self._surface)
            if not made_current:
                raise RuntimeError('Cannot make calibration OpenGL context current')
            target = np.asarray(target_px, dtype=float)
            live = None if live_px is None else np.asarray(live_px, dtype=float)
            faces = np.asarray(faces, dtype=int)
            if self._fbo is None or self._fbo.width() != width or self._fbo.height() != height:
                for samples in dict.fromkeys((self._samples, 0)):
                    fmt = QOpenGLFramebufferObjectFormat()
                    fmt.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
                    fmt.setInternalTextureFormat(0x8058)  # RGBA8
                    fmt.setSamples(samples)
                    self._fbo = QOpenGLFramebufferObject(width, height, fmt)
                    if self._fbo.isValid():
                        break
                    # Unsupported antialiasing should not reject an otherwise
                    # usable older GPU. Retry a plain depth framebuffer.
                    for _ in range(8):
                        if not self._gl.glGetError():
                            break
                self.info['samples'] = self._fbo.format().samples()
            if not self._fbo.isValid() or not self._fbo.bind():
                raise RuntimeError('Calibration framebuffer creation/bind failed')
            gl = self._gl
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(24. / 255., 31. / 255., 45. / 255., 1.)
            gl.glColorMask(True, True, True, True)
            gl.glDepthMask(True)
            gl.glClear(0x00004000 | 0x00000100)
            gl.glEnable(0x0b71)  # DEPTH_TEST
            gl.glEnable(0x809d)  # MULTISAMPLE (harmless for a plain framebuffer).
            if not self._program.bind():
                raise RuntimeError('Calibration shader bind failed')
            depths = target[:, 2] if live is None else np.r_[target[:, 2], live[:, 2]]
            low, high = float(depths.min()), float(depths.max())
            scale = 1.8 / max(high - low, 1.)
            self._program.setUniformValue(self._uniforms['u_viewport'], QVector2D(width, height))
            self._program.setUniformValue(self._uniforms['u_zmap'], QVector2D((high + low) * .5, scale))
            self._program.setUniformValue(self._uniforms['u_target_range'], QVector2D(float(target[:, 2].min()), float(target[:, 2].max())))
            if live is not None:
                self._surface_pass(self._vertices(live, faces, target=False), target=False, bias=0.)
            span = max(float(np.ptp(target[:, 2])), float(np.ptp(live[:, 2])) if live is not None else 0., 1.)
            self._surface_pass(self._vertices(target, faces, target=True), target=True, bias=span * 1e-5 * scale)
            self._buffer.release()
            self._program.release()
            gl.glDepthMask(True)
            error = gl.glGetError()
            if error:
                raise RuntimeError(f'Calibration OpenGL draw failed (0x{error:x})')
            image = self._fbo.toImage()
            self._fbo.release()
            if image.isNull() or gl.glGetError():
                raise RuntimeError('Calibration OpenGL readback failed')
            return image
        finally:
            if made_current:
                self._context.doneCurrent()
            if previous is not None and previous.isValid() and previous_surface is not None:
                previous.makeCurrent(previous_surface)

    def close(self):
        context = self._context
        if context is None:
            return
        previous = QOpenGLContext.currentContext()
        previous_surface = previous.surface() if previous is not None else None
        current = context.isValid() and self._surface is not None and context.makeCurrent(self._surface)
        try:
            if current:
                self._fbo = None
                if self._buffer is not None:
                    self._buffer.destroy()
        finally:
            self._fbo = self._buffer = self._program = None
            if current:
                context.doneCurrent()
            if self._surface is not None:
                self._surface.destroy()
            self._surface = self._context = None
            if (previous is not None and previous is not context and previous.isValid()
                    and previous_surface is not None):
                previous.makeCurrent(previous_surface)
