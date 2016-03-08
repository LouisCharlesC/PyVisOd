from PyQt4 import QtCore, QtGui
from Visualization import PyQt4OpenGlWidget
import numpy as np
from Visualization.PyQt4OpenGlWidget import GLPlotWidget

class MyQlabel(QtGui.QLabel):
	def __init__(self, parent):
		QtGui.QLabel.__init__(self, parent)
	
	def set_other_qlabel(self, qlabel):
		self._other_qlabels = qlabel
		
	def draw_target(self, u, v):
		painter = QtGui.QPainter(self.pixmap())
		painter.setPen(QtGui.QPen(QtGui.QBrush(QtGui.QColor(255,0,0)),2))
		painter.drawLine(QtCore.QPoint(u,0), QtCore.QPoint(u,self.pixmap().height()))
		painter.drawLine(QtCore.QPoint(0,v), QtCore.QPoint(self.pixmap().width(),v))
		self.repaint()
		del painter

	def mousePressEvent(self, event):
		if event.buttons() & QtCore.Qt.LeftButton:
			self.draw_target(event.pos().x(), event.pos().y())
			for qlabel in self._other_qlabels:
				qlabel.draw_target(event.pos().x(), event.pos().y())

class Tutorial(QtGui.QWidget):
	def __init__(self, cb, closed, paused):
		super(Tutorial, self).__init__()
		self._close = closed
		self._paused = paused

		self._nb_param = 6
		wnd_sz_w = 1580
		wnd_sz_h = 950
		self._img_sz_w = 640
		self._img_sz_h = 480
		self._img_sz_ratio = 0.9
		self.setGeometry(50, 50, wnd_sz_w, wnd_sz_h)
		self.setWindowTitle('Visual Monocular Scene Reconstruction')
		self._grad_color_table = [QtGui.qRgb(val,val,val) for val in xrange(256)]

		hbox = QtGui.QHBoxLayout()
		self.setLayout(hbox)
		vbox_rgb = QtGui.QVBoxLayout()
		vbox_gl = QtGui.QVBoxLayout()
		hbox.addLayout(vbox_rgb)
		hbox.addLayout(vbox_gl)
		hbox.addStretch(1)
		# RGB images
		self._label_rgb_from = MyQlabel(self)
		self._label_rgb_from.setMinimumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		self._label_rgb_from.setMaximumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		self._label_rgb_to = MyQlabel(self)
		self._label_rgb_to.setMinimumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		self._label_rgb_to.setMaximumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		self._label_rgb_from.set_other_qlabel([self._label_rgb_to])
		self._label_rgb_to.set_other_qlabel([self._label_rgb_from])
		vbox_rgb.addWidget(self._label_rgb_from)
		vbox_rgb.addWidget(self._label_rgb_to)
		vbox_rgb.addStretch(1)
		# OpenGL
		self._opengl = GLPlotWidget(self)
		self._opengl.setMinimumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		self._opengl.setMaximumSize(QtCore.QSize(self._img_sz_w*self._img_sz_ratio,self._img_sz_h*self._img_sz_ratio))
		vbox_gl.addWidget(self._opengl)
		vbox_gl.addStretch(1)

		btn = QtGui.QPushButton('Process next image', self)
		btn.resize(btn.sizeHint())
		btn.move(wnd_sz_w-btn.width(), wnd_sz_h-btn.height())
		if not cb == None:
			btn.clicked.connect(cb)
		
		self.show()
		
	def keyPressEvent(self, event):
		if event.key() == QtCore.Qt.Key_P:
			self._paused[0] = not self._paused[0]
			print "Pause: " + str(self._paused[0])
			event.accept()
		
	def closeEvent(self, event):
		del self._label_rgb_from, self._label_rgb_to
		self._close.remove(False)
		self._close.append(True)
		event.accept()
	
	def update_images(self, rgb_from, rgb_to, ptcld):
		pixmap_from = QtGui.QPixmap.fromImage(QtGui.QImage(rgb_from.data,self._img_sz_w,self._img_sz_h,QtGui.QImage.Format_RGB888))
		self._label_rgb_from.setPixmap(pixmap_from.scaled(self._label_rgb_from.size(), QtCore.Qt.KeepAspectRatio))
		self._label_rgb_from.repaint()
		
		pixmap_to = QtGui.QPixmap.fromImage(QtGui.QImage(rgb_to.data,self._img_sz_w,self._img_sz_h,QtGui.QImage.Format_RGB888))
		self._label_rgb_to.setPixmap(pixmap_to.scaled(self._label_rgb_to.size(), QtCore.Qt.KeepAspectRatio))
		self._label_rgb_to.repaint()
		
		self._opengl.set_data(ptcld)
		self._opengl.updateGL()
		
		