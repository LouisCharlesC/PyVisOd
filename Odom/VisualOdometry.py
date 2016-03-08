import numpy as np

from utils import image, GradientDescentOptimization

class VisualOdometry:		
	def __init__(self, image_from, nb_it_max=50):
		self._n_p = 6
		#---------------------------------------------------------------- Parameters
		self._color_grad_min = np.percentile(np.max(np.abs(image_from.get_color_gradients()), axis=2), 95)
		self._angle_min_tan = 0.#np.tan(np.pi/6.)
		self._nb_it_max = nb_it_max
		#---------------------------------------------------------- Save useful info
		self._f_x = image_from._f_x
		self._f_y = image_from._f_y
		self._c_x = image_from._c_x
		self._c_y = image_from._c_y
		self._width = image_from._color.shape[1]
		self._height = image_from._color.shape[0]
		self._nbr_color_chan = image_from._color.shape[2]
		
		#------------------------------------------------- Select valid input points
		validity_mask = self._compute_validity_mask(image_from)
		print np.count_nonzero(validity_mask)
		self._ref = np.compress(np.ravel(validity_mask), np.reshape(image_from._color, [-1,self._nbr_color_chan]), axis=0)
		self._xyz_ref = np.compress(np.ravel(validity_mask), np.reshape(image_from.get_ptcld()[...,:4], [-1,4]), axis=0)
		
#		#---------------------------------- Pre-compute jacobian and inverse Hessian
		J_from_gradperpix = np.compress(np.ravel(validity_mask), np.reshape(image_from.get_color_gradients(), [-1,self._nbr_color_chan,2]), axis=0)
		self._z_ref_inv = 1./self._xyz_ref[:,2]
		z_ref_sq_inv = np.square(self._z_ref_inv)
		J_ref_pixpertf = np.concatenate([np.concatenate([image_from._f_x*self._z_ref_inv[:,np.newaxis], 
																										 np.zeros_like(self._z_ref_inv[:,np.newaxis]), 
																										 -image_from._f_x*(self._xyz_ref[:,0]*z_ref_sq_inv)[:,np.newaxis], 
																										 -image_from._f_x*((self._xyz_ref[:,0]*self._xyz_ref[:,1])*z_ref_sq_inv)[:,np.newaxis], 
																										 image_from._f_x*(1.+np.square(self._xyz_ref[:,0])*z_ref_sq_inv)[:,np.newaxis], 
																										 -image_from._f_x*(self._xyz_ref[:,1]*self._z_ref_inv)[:,np.newaxis]]
																									  , axis=1)[:,:,np.newaxis],
										                 np.concatenate([np.zeros_like(self._z_ref_inv[:,np.newaxis]), 
																										 image_from._f_y*self._z_ref_inv[:,np.newaxis], 
																										 -image_from._f_y*(self._xyz_ref[:,1]*z_ref_sq_inv)[:,np.newaxis], 
																										 -image_from._f_y*(1.+np.square(self._xyz_ref[:,1])*z_ref_sq_inv)[:,np.newaxis], 
																										 image_from._f_y*((self._xyz_ref[:,0]*self._xyz_ref[:,1])*z_ref_sq_inv)[:,np.newaxis], 
																										 image_from._f_y*(self._xyz_ref[:,0]*self._z_ref_inv)[:,np.newaxis]]
																									  , axis=1)[:,:,np.newaxis]]
																	  , axis=2)

		J_ref_gradpertf = np.sum(np.multiply(J_from_gradperpix[:,:,np.newaxis,:],J_ref_pixpertf[:,np.newaxis,:,:]), axis=3)
		
		self._J_ref = np.reshape(J_ref_gradpertf, [-1,self._n_p])
		self._J_ref_pixpertf_max = np.max(np.max(np.abs(J_ref_pixpertf), axis=2), axis=0)[:,np.newaxis]
		u, s, v = np.linalg.svd(self._J_ref, full_matrices=False)
		#-------------------- Pre-compute some transposes and squares for future use
		self._ref_u_t = np.transpose(u)
		self._ref_s = s[:,np.newaxis]
		self._ref_s_sq = np.square(self._ref_s)
		self._ref_v_t = np.transpose(v)
		
	def get_error(self, cur, w):
		return np.sum(np.square(np.multiply(w, self._get_residuals(cur)))) / max(np.sum(w), np.finfo(np.float).eps)
	
	def _get_residuals(self, cur):
		return cur - self._ref
		
	def _get_jacobian(self, beta):
		return np.dot(self._ref_v_t, np.multiply(self._ref_s/(self._ref_s_sq+beta), np.multiply(self._ref_u_t, np.reshape(self._w, [1,-1]))))
	
	def _apply_step(self, tf, delta_lie):
		#------------------------ Transform into delta_lie into tranformation matrix
		theta = np.linalg.norm(delta_lie[3:])
		if theta**3 <= np.finfo(np.float).tiny:
			delta_tf = np.vstack([np.hstack([np.eye(3, 3, 0, dtype=np.float), delta_lie[:3]]), np.array([0., 0., 0., 1.], dtype=np.float)])
		else:
			wx = np.array([[0., -delta_lie[5], delta_lie[4]], [delta_lie[5], 0., -delta_lie[3]], [-delta_lie[4], delta_lie[3], 0.]], dtype=np.float)
			wx_2 = np.linalg.matrix_power(wx,2)
			const_1 = (1-np.cos(theta))/theta**2
			e_wx = np.eye(3, 3, 0, dtype=np.float) + np.sin(theta)*wx/theta + const_1*wx_2
			V = np.eye(3, 3, 0, dtype=np.float) + const_1*wx + (theta - np.sin(theta))*wx_2/theta**3
			delta_tf = np.vstack([np.hstack([e_wx, np.dot(V, delta_lie[:3])]), np.array([0., 0., 0., 1.], dtype=np.float)])
		return self._apply_delta_tf(tf, delta_tf)
			
	def _apply_delta_tf(self, tf, delta_tf):
		#------------------------------------------------------------ Compose two tf
		return np.dot(delta_tf, tf)
	
	def _limit_tf_step(self, delta_lie):
		#--------------- Scale delta_lie so that no point moves by more than 1 pixel
		factor = np.max(np.abs(delta_lie) * self._J_ref_pixpertf_max)
		if factor > 1.:
			print ',',
			delta_lie /= factor
		return delta_lie
		
	def _compute_validity_mask(self, image):
		# NOTE: this should be a color-channel wise selection
		#-------------------------------------------------- Only select points with:
		# Depth values
		self._valid_depth_mask = np.not_equal(image.get_ptcld()[...,2], -1.)
		# Sufficient color gradient
		valid_color_grad = np.any(np.greater(np.max(np.abs(image.get_color_gradients()), axis=2), self._color_grad_min), axis=2)
		# Low angle wrt camera
		depth_grad_mag = np.sqrt(np.sum(np.square(image.get_depth_gradient()), axis=2))
		valid_depth_angle = np.greater_equal(image.get_ptcld()[...,2]/(2./(1./image._f_x+1./image._f_y))/np.where(depth_grad_mag==0., np.finfo(np.float).eps, depth_grad_mag), self._angle_min_tan)
		#-------------------------------------------------- Merge the above criteria
		return np.logical_and(np.logical_and(self._valid_depth_mask, valid_color_grad), valid_depth_angle)
	
	def apply_inv_tf(self, tf, ptcld):
		#----------------------------------------------------------------- Invert tf
		R_t = np.transpose(tf[:3,:3])
		tf_inv = np.vstack([np.hstack([R_t, np.dot(-R_t, tf[:3,3][:,np.newaxis])]), np.array([0., 0., 0., 1.], dtype=np.float)])
		#------------------------------------------- Compute delta idx for each pixel
		pt_cld_tf = np.tensordot(ptcld, tf_inv, axes=([-1],[1]))
		idx_tf = np.concatenate([(self._f_y*pt_cld_tf[...,1]/pt_cld_tf[...,2]+self._c_y)[np.newaxis,...], (self._f_x*pt_cld_tf[...,0]/pt_cld_tf[...,2]+self._c_x)[np.newaxis,...]], axis=0)
#		idx_tf = np.add(self._idx, delta_idx)
		idx_tf_dec = np.round(idx_tf)
		idx_tf_mod = np.mod(idx_tf_dec, np.reshape(np.array([self._height, self._width],dtype=np.int32), [2,]+[1]*(idx_tf_dec.ndim-1)))
		idx_tf_int = idx_tf_mod.astype(np.int32)
		idx_tf_frac = idx_tf-idx_tf_dec
		#---------------------------------------------- Identify out-of-image points
		w_out = np.repeat(np.all(idx_tf_mod == idx_tf_dec, axis=0).astype(np.float)[...,np.newaxis], self._nbr_color_chan, -1)
		#-------------------------------------------------- Warp the reference image
		return np.add(self._cur[idx_tf_int[0],idx_tf_int[1]],np.add(np.multiply(idx_tf_frac[0,...,np.newaxis], self._J_cur_gradperpix[idx_tf_int[0],idx_tf_int[1],:,1]),np.multiply(idx_tf_frac[1,...,np.newaxis], self._J_cur_gradperpix[idx_tf_int[0],idx_tf_int[1],:,0]))), w_out

	def align(self, image, tf_init=np.eye(4, 4, 0, dtype=np.float)):
		#-------------------------------- Run the gradient descent until convergence
		self._cur = image._color
		self._J_cur_gradperpix = image.get_color_gradients()
		self._tf = tf_init
		self._cur_tf, self._w = self.apply_inv_tf(self._tf, self._xyz_ref)

		opt = GradientDescentOptimization.GradientDescentOptimization(self._get_residuals, self._get_jacobian, np.array([0.001,0.001,0.001,0.001,0.001,0.001], dtype=np.float), 5, self._nb_it_max)
		best_error = self.get_error(self._cur_tf, self._w)
		beta = np.mean(np.abs(self._ref_s))#1024.
		nb_it = 0
#		print best_error
		while not opt.get_converged():
			tf_step = opt.run_single_step(self._cur_tf, beta)
			tf_step = self._limit_tf_step(tf_step)
			tf_tmp = self._apply_step(self._tf, tf_step)
			tmp, w = self.apply_inv_tf(tf_tmp, self._xyz_ref)
			error_tmp = self.get_error(tmp, w)
				
			if error_tmp < best_error:
				self._tf = tf_tmp
				self._cur_tf = tmp
				self._w = w
				beta *= 2.
			
				best_error = error_tmp
				print ' ',
			else:
				beta /= 32.#np.sqrt(beta)
				print '.',

			_, nb_it = opt.check_convergence(tf_step)
			
		return self._tf, nb_it


if __name__ == '__main__':
	#--------------------------------------------------------- Set-up simple tests
	img_ref = np.array([[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]], dtype=np.int)
	d_ref = np.ones([img_ref.shape[0],img_ref.shape[1]], dtype=np.int)
	img_cur = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]], dtype=np.int)
#	img_ref = np.repeat(np.arange(200)[:,np.newaxis], 150, 1)+np.arange(0,2*150,2)[np.newaxis,:]
#	d_ref = np.ones([img_ref.shape[0],img_ref.shape[1]], dtype=np.int)
#	img_cur = np.repeat(np.arange(1,201)[:,np.newaxis], 150, 1)+np.arange(2,2*151,2)[np.newaxis,:]
#	img_ref = np.array([[10,9,8,7,4,5,6],[10,9,8,7,4,5,6],[10,9,8,7,4,5,6],[10,9,8,7,4,5,6],[10,9,8,7,4,5,6],[10,9,8,7,4,5,6]], dtype=np.int)
#	d_ref = np.array([[2.,2.,2.,2.,1.,1.,1.],[2.,2.,2.,2.,1.,1.,1.],[2.,2.,2.,2.,1.,1.,1.],[2.,2.,2.,2.,1.,1.,1.],[2.,2.,2.,2.,1.,1.,1.],[2.,2.,2.,2.,1.,1.,1.]], dtype=np.float64)*10.
#	img_cur = np.array([[9,8,4,5,6,7,8],[9,8,4,5,6,7,8],[9,8,4,5,6,7,8],[9,8,4,5,6,7,8],[9,8,4,5,6,7,8],[9,8,4,5,6,7,8]], dtype=np.int)
	c_x = (img_ref.shape[1]-1.)/2.
	c_y = (img_ref.shape[0]-1.)/2.
	f_x = 1.
	f_y = 1.
	
	image_from = image.image(img_ref[:,:,np.newaxis], d_ref, c_x, c_y, f_x, f_y)
	image_to = image.image(img_cur[:,:,np.newaxis], d_ref, c_x, c_y, f_x, f_y)
	vis_od = VisualOdometry(image_from)
	tf, nb_it = vis_od.align(image_to, np.eye(4, 4, 0, dtype=np.float))
	res, w = vis_od.apply_inv_tf(tf, image_from.get_ptcld()[...,:4])
	
	print img_cur
	print img_ref
	print res[...,0]*w[...,0]
	print tf
	print nb_it