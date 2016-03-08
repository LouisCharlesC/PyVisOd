import sys
#sys.setrecursionlimit(2048)
from time import clock, sleep

import numpy as np
from PyQt4 import QtGui

from utils.utils import inv_rot, quat_to_mat, mat_to_quat
from utils.image import image
from InputGrabber import ExtTimeDrivenTUMGrabber
from GUI import Tutorial
from Odom.VisualOdometry import VisualOdometry
from Octree import Octree

closed = [False]
paused = [False]
db_folder="../../db/rgbd_dataset_freiburg1_xyz"
nb_it_max = 200

def process_and_refresh():
	global image_from, gt_rot_from, ptcld, tf
	
	tf_cur_prev = np.eye(4, 4, 0, dtype=np.float)
	tf_cur_last = np.eye(4, 4, 0, dtype=np.float)
	with open(db_folder+"/res.txt", 'w') as ff:
		#---------------------------------------------------------- Process next image
		while not closed[0]:
			rgb_to, depth_to, gt_quat_to, stamp = grabber.get_next()
			gt_rot_to = quat_to_mat(gt_quat_to)
			image_to = image(rgb_to, depth_to, c_x, c_y, f_x, f_y)
			
			#---------------------------------------------------------------- Localize
			time_i = clock()
			vis_od = VisualOdometry(image_from, nb_it_max)
			tf_cur, nb_it = vis_od.align(image_to, tf_cur_prev)#np.dot(inv_rot(gt_rot_from), gt_rot_to))#np.eye(4, 4, 0, dtype=np.float64))
			time_f = clock()
			print "Time for \"localize\": " + str(time_f-time_i) + " seconds, " + str(nb_it) + " iterations."
			
			#--------------------------------------------------------------- Display stuff	
			rgb_tf, w = vis_od.apply_inv_tf(tf_cur, image_from.get_ptcld()[...,:4])
			rgb_tf = np.where(np.logical_or(image_from.get_ptcld()[...,2,np.newaxis]==-1., w==0.), image_from._color, rgb_tf)
			viz.update_images(np.ascontiguousarray(image_from._color, dtype=np.uint8), np.ascontiguousarray(rgb_tf, dtype=np.uint8), ptcld.leaf_data.get_data())
			
			#------------------------------------------------------------ If converged
			if nb_it < nb_it_max:
				tf_cur_prev = tf_cur
				tf_cur_last = tf_cur
				#------------------------------------------------ Apply current estimate
				tf = np.dot(tf, tf_cur)
				tf_quat = mat_to_quat(tf)
				#------------------------------------------------------- Save trajectory
				ff.write(str(stamp)+' '+str(tf_quat[0])+' '+str(tf_quat[1])+' '+str(tf_quat[2])+' '+str(tf_quat[3])+' '+str(tf_quat[4])+' '+str(tf_quat[5])+' '+str(tf_quat[6])+'\n')
				
				#-------------------------------------------------------- Estimate depth
				
			
				#------------------------------------------------------ Integrate point clouds
				time_i = clock()
				ptcld_to = image_to.get_valid_ptcld()
				ptcld_to[:,:4] = np.tensordot(ptcld_to[:,:4], tf, axes=([-1],[1]))
				ptcld.insert(ptcld_to)
				time_f = clock()
				print "Time for \"voxelize\": " + str(time_f-time_i) + " seconds."
			
			#---------------------------------------------------- Get ready for next image
			image_from = image_to
			gt_rot_from = gt_rot_to
#			else:
#				tf_cur_prev = vis_od._apply_delta_tf(tf_cur_prev, tf_cur_last)
				
			app.processEvents()
			while (paused[0] or not grabber.peek_next()) and not closed[0]:
				sleep(0.1)
				app.processEvents()

if __name__ == '__main__':
	global grabber, image_from, gt_rot_from, c_x, c_y, f_x, f_y, ptcld, tf, app, viz
	
	# Get initial data
	grabber = ExtTimeDrivenTUMGrabber.ExtTimeDrivenTUMGrabber(db_folder=db_folder)
	rgb_from, depth_from, gt_quat_from, _ = grabber.get_next()
	c_x, c_y, f_x, f_y = grabber.get_intrinsic_parameters()
	
	# Make ref image
	gt_rot_from = quat_to_mat(gt_quat_from)
	image_from = image(rgb_from, depth_from, c_x, c_y, f_x, f_y)
	ptcld_from = image_from.get_valid_ptcld()
	tf = gt_rot_from
	ptcld_from[:,:4] = np.tensordot(ptcld_from[:,:4], tf, axes=([-1],[1]))
	ptcld = Octree.Octree(ptcld_from, np.array([-5.,-5.,-5.]), np.array([5.,5.,5.]), .02)
	
	#------------------------------------ Visualization last, it's a blocking call
	app = QtGui.QApplication(sys.argv)
	viz = Tutorial.Tutorial(cb=process_and_refresh, closed=closed, paused=paused)
	sys.exit(app.exec_())
	
	