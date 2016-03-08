import numpy as np

class GradientDescentOptimization():
	def __init__(self, get_residuals_func, get_jacobian_func, crit_step, crit_step_it, max_it):
		self._get_residuals = get_residuals_func
		self._get_jacobian = get_jacobian_func
		
		self._crit_step = crit_step
		self._crit_step_it = crit_step_it
		self._max_it = max_it
		
		self._it = 0
		self._crit_it = 0
		self.check_convergence(np.zeros((6,1), dtype=np.float64))
		
	def run_single_step(self, tf, beta):
		res = np.reshape(self._get_residuals(tf), [-1,1])
		J_inv = self._get_jacobian(beta)
		
		self._it += 1
		return np.dot(J_inv,res)
	
	def check_convergence(self, step):
		if np.any(np.abs(step) > self._crit_step):
			self._crit_it=0
		else:
			self._crit_it+=1
		self._converged = self._it >= self._max_it or self._crit_it > self._crit_step_it
		return self._converged, self._it
	
	def get_converged(self):
		return self._converged
	
if __name__ == "__main__":
	from time import clock
	global tf
	# imagine a function: x^4-2*(x^3)-x^2+x
	# add some parameters: b4^2*(x^4)+b3*(x^3)+b2*(x^2)+b1*(x)+b0
	# use gn opt to find param values
	
	# create 20 samples at random
	nb_samples = 20000
	xvals = np.random.uniform(-2.,2.,[nb_samples,1])
	yvals = np.power(xvals,4)-np.multiply(2,np.power(xvals,3))-np.power(xvals,2)+xvals
	
	def residuals(tf):
		return (np.multiply(np.power(tf[4],4),np.power(xvals,4))+np.multiply(tf[3],np.power(xvals,3))+np.multiply(tf[2],np.power(xvals,2))+np.multiply(tf[1],xvals)+tf[0]) - yvals
	
	def jacobian(beta):
		global tf
		J = np.hstack((np.ones([nb_samples,1]), xvals, np.power(xvals,2), np.power(xvals,3), 2*tf[4]*np.power(xvals,4)))
		u, s, v = np.linalg.svd(J, full_matrices=False)
		return np.dot(np.transpose(v), np.multiply(s[:,np.newaxis]/(np.square(s[:,np.newaxis])+beta), np.transpose(u)))
	
	def apply_step(best, bstep):
		return best-bstep
	
	tf = np.random.random([5,1])
	best_error = sum(np.square(residuals(tf)))
	best_tf = tf
	opt = GradientDescentOptimization(residuals, jacobian, 0.001, 5, 200)
	print best_error
	beta = 0.001
	time_i = clock()
	while not opt.get_converged():
		tf_step = opt.run_single_step(tf, beta)
		tf_tmp = apply_step(tf, tf_step)
		error_tmp = sum(np.square(residuals(tf_tmp)))
		if error_tmp < best_error:
			tf = tf_tmp
			beta /= 10.
			
			opt.check_convergence(tf_step)
			
			if error_tmp < best_error:
				best_tf = tf
				best_error = error_tmp
				print best_error
		else:
			beta *= 10.
		
	print best_tf, beta
	print clock()-time_i
	
	