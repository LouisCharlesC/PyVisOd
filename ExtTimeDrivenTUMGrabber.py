from InputGrabber import InputGrabber
from scipy import misc, array
import numpy as np
from string import split

class ExtTimeDrivenTUMGrabber(InputGrabber):
	def __init__(self, time=0., db_folder="./"):
		InputGrabber.__init__(self, time)
		self._db_folder = db_folder.rstrip("/") + "/"
		
		#----------------------------------------------------------- Load rgb images
		with open(self._db_folder+"/rgb.txt") as ff:
			line = ff.readline()
			while line.startswith('#'):
				line = ff.readline()
			self._rgb_lines = [[line.split(' ')[0], line.split(' ')[1].rstrip('\n')]]
			self._rgb_lines.extend([[line.split(' ')[0], line.split(' ')[1].rstrip('\n')] for line in ff.readlines()])
		self._rgb_len = len(self._rgb_lines)
		self._rgb_ptr = 0
		
		#--------------------------------------------------------- Load depth images
		with open(self._db_folder+"/depth.txt") as ff:
			line = ff.readline()
			while line.startswith('#'):
				line = ff.readline()
			self._depth_lines = [[line.split(' ')[0], line.split(' ')[1].rstrip('\n')]]
			self._depth_lines.extend([[line.split(' ')[0], line.split(' ')[1].rstrip('\n')] for line in ff.readlines()])
		self._depth_len = len(self._depth_lines)
		self._depth_ptr = 0
		
		#--------------------------------------------------------- Load ground truth
		with open(self._db_folder+"/groundtruth.txt") as ff:
			line = ff.readline()
			while line.startswith('#'):
				line = ff.readline()
			self._gt_lines = [[line.split(' ')[0], line.rstrip('\n').split(' ')[1:]]]
			self._gt_lines.extend([[float(line.split(' ')[0]), line.rstrip('\n').split(' ')[1:]] for line in ff.readlines()])
		self._gt_len = len(self._gt_lines)
		self._gt_ptr = 1
		
		#---------------------------------------------------------- Set current time
		self._time = max(self._time, self._rgb_lines[self._rgb_ptr][0])
		print "Grabber init done."
		
	def peek_next(self):
		return self._rgb_ptr < self._rgb_len and self._depth_ptr < self._depth_len and self._gt_ptr < self._gt_len
	
	def get_next(self):
		rgb, stamp = self.get_rgb()
		depth, _ = self.get_depth()
		gt = self.get_gt(stamp)
		return rgb, depth, gt, stamp
		
	def get_rgb(self):
		'''Return the next rgb image from the database'''
		if self._rgb_ptr < self._rgb_len:
			img = misc.imread(self._db_folder+self._rgb_lines[self._rgb_ptr][1])
			stamp = float(self._rgb_lines[self._rgb_ptr][0])
			self._rgb_ptr += 1
		else:
			img = None
			stamp = self._time
		return img, stamp
	
	def get_depth(self):
		'''Return the next depth image from the database'''
		if self._depth_ptr < self._depth_len:
			depth = misc.imread(self._db_folder+self._depth_lines[self._depth_ptr][1]).astype(float)/5000.
			depth = np.where(depth==0., -1., depth)
			stamp = float(self._depth_lines[self._depth_ptr][0])
			self._depth_ptr += 1
		else:
			depth = None
			stamp = self._time
		return depth, stamp
	
	def get_gt(self, stamp_in):
		'''Return the ground truth position associated to the time stamp'''
		gt = None
		while self._gt_ptr < self._gt_len:
			stamp_diff = float(self._gt_lines[self._gt_ptr][0]) - stamp_in
			if stamp_diff >= 0:
				stamp_diff_prev = stamp_in - float(self._gt_lines[self._gt_ptr-1][0])
				gt_prev = array([float(nbr)*stamp_diff for nbr in self._gt_lines[self._gt_ptr-1][1]])
				gt = (array([float(nbr)*stamp_diff_prev for nbr in self._gt_lines[self._gt_ptr][1]])+gt_prev)/(stamp_diff+stamp_diff_prev)
				self._gt_ptr+=1
				break
			self._gt_ptr+=1
		return gt
		
	def get_intrinsic_parameters(self):
		c_x = 318.5
		c_y = 255.3
		f_x = 517.3
		f_y = 516.5
		return c_x, c_y, f_x, f_y
#if __name__ == "__main__":
#	ExtTimeDrivenTUMGrabber(db_folder="../../db/rgbd_dataset_freiburg1_xyz")