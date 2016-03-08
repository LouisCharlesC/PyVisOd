# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import numpy as np

def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader

def compile_fragment_shader(source):
    """Compile a fragment shader from source."""
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, source)
    gl.glCompileShader(fragment_shader)
    # check compilation error
    result = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(fragment_shader))
    return fragment_shader

def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program with from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

# Vertex shader
VS = """
#version 120
varying vec4 vertex_color;
void main()
{
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  vertex_color = gl_Color;
}
"""

# Fragment shader
FS = """
#version 120
varying vec4 vertex_color;
void main()
{
    gl_FragColor = vertex_color;
}
"""

class GLPlotWidget(QGLWidget):
    def __init__(self, parent=None):
      super(GLPlotWidget, self).__init__(parent)
      
      self.xRot = 0.
      self.yRot = 0.
      self.zRot = 0.
      self.x = 0.
      self.y = 0.
      self.z = 0.
      
      self.lastPos = QtCore.QPoint()
      
    def __del__(self):
      gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
      gl.glDisableClientState(gl.GL_COLOR_ARRAY)
      self.vbo.unbind()
      self.veo.unbind()
      gl.glUseProgram(0)

    def initializeGL(self):
      """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
      # background color
      gl.glClearColor(0, 0, 0, 0)
      # create a Vertex Buffer Object with the specified data
##      elements = np.repeat(np.repeat(np.arange(0,w*(h-1),w,dtype=np.int32), w-1+w-1)[:,np.newaxis], 3, axis=1) + np.tile(np.hstack([np.repeat(np.arange(w-1,dtype=np.int32), 2)[:,np.newaxis], np.repeat(np.arange(1+w,w+w,dtype=np.int32), 2)[:,np.newaxis], np.reshape(np.hstack([np.arange(w, w+w-1,dtype=np.int32)[:,np.newaxis], np.arange(1,w,dtype=np.int32)[:,np.newaxis]]), [-1,1])]), [h-1,1])
##      self.veo = glvbo.VBO(elements, target=gl.GL_ELEMENT_ARRAY_BUFFER)
##      self.veo.bind()
      self._len_data = 1
      self.vbo = glvbo.VBO(np.empty([self._len_data,6],np.float32), target=gl.GL_ARRAY_BUFFER)
      self.vbo.bind()
      
      # compile the vertex shader
      vs = compile_vertex_shader(VS)
      # compile the fragment shader
      fs = compile_fragment_shader(FS)
      # compile the vertex shader
      self.shaders_program = link_shader_program(vs, fs)
        
      gl.glUseProgram(self.shaders_program)
      
    def set_data(self, data):
      self._len_data = data.shape[0]
      self.vbo.unbind()
      self.vbo = glvbo.VBO(data, target=gl.GL_ARRAY_BUFFER)
      self.vbo.bind()

    def paintGL(self):
      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      
      gl.glMatrixMode(gl.GL_MODELVIEW)
      gl.glLoadIdentity()
      gl.glRotatef(self.xRot, 1.0, 0.0, 0.0)
      gl.glRotatef(self.yRot, 0.0, 1.0, 0.0)
      gl.glRotatef(self.zRot, 0.0, 0.0, 1.0)
      gl.glTranslatef(self.x, self.y, self.z)
      
      gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
      gl.glEnableClientState(gl.GL_COLOR_ARRAY)
      
      gl.glVertexPointer(3, gl.GL_FLOAT, 32, self.vbo)
      gl.glColorPointer(3, gl.GL_FLOAT, 32, self.vbo+16)
      
#      gl.glDrawElements(gl.GL_TRIANGLES, 24, gl.GL_UNSIGNED_INT, None)
      gl.glDrawArrays(gl.GL_POINTS, 0, self._len_data)

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
#        gl.glFrustum(1., -1., -1., 1., 0.5, 5.)
        gl.glOrtho(-1., 1., 1., -1., -0.1, -10.)
        
    def setXRotation(self, angle):
      angle = self.normalizeAngle(angle)
      if angle != 0.:
        self.xRot += angle

    def setYRotation(self, angle):
      angle = self.normalizeAngle(angle)
      if angle != 0.:
        self.yRot += angle

    def setZRotation(self, angle):
      angle = self.normalizeAngle(angle)
      if angle != 0.:
        self.zRot += angle
            
    def normalizeAngle(self, angle):
      while angle < 0.:
        angle += 360.
      while angle >= 360.:
        angle -= 360.
      return angle
    
    def wheelEvent(self, event):
      self.z += event.delta()/1200.
      self.updateGL()
      event.accept()
        
    def mousePressEvent(self, event):
      self.lastPos = event.pos()
      event.accept()
      
    def mouseDoubleClickEvent(self, event):
      self.xRot = 0.
      self.yRot = 0.
      self.zRot = 0.
      self.x = 0.
      self.y = 0.
      self.z = 0.
      self.updateGL()
      event.accept()

    def mouseMoveEvent(self, event):
      if event.buttons() & (QtCore.Qt.LeftButton | QtCore.Qt.RightButton | QtCore.Qt.MidButton):
        du = -float(event.x() - self.lastPos.x())
        dv = -float(event.y() - self.lastPos.y())
  
        if event.buttons() & QtCore.Qt.LeftButton:
          self.yRot = self.normalizeAngle(self.yRot + du*0.5)
          self.xRot = self.normalizeAngle(self.xRot + dv*0.5)
        elif event.buttons() & QtCore.Qt.RightButton:
          self.yRot = self.normalizeAngle(self.yRot + du*0.5)
          self.zRot = self.normalizeAngle(self.zRot + dv*0.5)
        elif event.buttons() & QtCore.Qt.MidButton:
          self.x += du/120.
          self.y += dv/120.
        
        if (du != 0.) or (dv != 0.):
          self.updateGL()

        self.lastPos = event.pos()
        event.accept()
