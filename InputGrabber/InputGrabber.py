
class InputGrabber:
	def __init__(self, time=0.):
		self._time = time
		
	def get_input(self):
		print "InputGrabber::get_input() called but not overloaded."
	
	def set_time(self, time):
		self._time = time