'''
Created on Feb 2, 2016

@author: lcc
'''
import numpy as np
from scipy import ndimage

class image:
  def __init__(self, color, depth, c_x, c_y, f_x, f_y):
    self._color = color.astype(np.float)
    self._depth = depth.astype(np.float)
    self._c_x = c_x
    self._c_y = c_y
    self._f_x = f_x
    self._f_y = f_y
    self._ptcld = None
    self._color_gradients = None
    self._depth_gradient = None
    
  def get_color_gradients(self):
    if self._color_gradients == None:
      self._color_gradients = np.concatenate([self._compute_gradient(self._color[:,:,chan])[:,:,np.newaxis,:] for chan in xrange(self._color.shape[2])], axis=2)
    return self._color_gradients
  
  def get_depth_gradient(self):
    if self._depth_gradient == None:
      self._depth_gradient = self._compute_gradient(self._depth)
    return self._depth_gradient
      
  def _compute_gradient(self, img):
    x_kernel = np.array([[3., 0., -3.],[10, 0., -10.],[3., 0., -3.]], dtype=np.float)
    y_kernel = np.array([[3., 10., 3.],[0, 0., 0.],[-3., -10., -3.]], dtype=np.float)
    x_kernel = x_kernel / np.sum(np.abs(x_kernel))
    y_kernel = y_kernel / np.sum(np.abs(y_kernel))
    
    return np.concatenate([ndimage.filters.convolve(img,x_kernel,mode='nearest')[:,:,np.newaxis], 
                           ndimage.filters.convolve(img,y_kernel,mode='nearest')[:,:,np.newaxis]], axis=2)
    
  def get_ptcld(self):
    if self._ptcld == None:
      self._ptcld = self._make_ptcld(self._color, self._depth, self._c_x, self._c_y, self._f_x, self._f_y)
    return self._ptcld
  
  def get_valid_ptcld(self):
    if self._ptcld == None:
      self._ptcld = self._make_ptcld(self._color, self._depth, self._c_x, self._c_y, self._f_x, self._f_y)
    valid_mask = np.ravel(np.greater(self._ptcld[...,2], 0.))
    return np.compress(valid_mask, np.reshape(self._ptcld, [-1, 8]), axis=0)
    
  def _make_ptcld(self, color, depth, c_x, c_y, f_x, f_y):
    basis_u = np.multiply(np.subtract(np.reshape(np.arange(color.shape[1],dtype=np.float), [1,-1]), c_x), 1./f_x)
    basis_v = np.multiply(np.subtract(np.reshape(np.arange(color.shape[0],dtype=np.float), [-1,1]), c_y), 1./f_y)
    return np.concatenate([np.multiply(basis_u, depth)[:,:,np.newaxis], 
                           np.multiply(basis_v, depth)[:,:,np.newaxis], 
                           depth[:,:,np.newaxis], 
                           np.ones_like(depth)[:,:,np.newaxis], 
                           color/255.,
                           np.ones_like(depth)[:,:,np.newaxis]]
                          , axis=2)