'''
Created on Feb 12, 2016

@author: lcc
'''

#TODO: breadth-first construction = fewer cache-misses
#      make it a N^3 tree rather ? so that larger groups are pushed down fewer levels of the tree

import numpy as np

#===============================================================================
# A non-existent node, implements merging
#===============================================================================
class OctreeEmpty:
  def merge(self, data, data_leaf_coord, start_idx, end_idx, depth, leaf_data):
    return Octree._makeNewNode(data, data_leaf_coord, start_idx, end_idx, depth, leaf_data)

#===============================================================================
# A lead, contains data
#===============================================================================
class OctreeLeaf:
  def __init__(self, data, data_leaf_coord, start_idx, end_idx, leaf_data):
    self._data = np.mean(data[start_idx:end_idx], axis=0)[np.newaxis,:]
    self._data_leaf_coord = data_leaf_coord[start_idx]
    self._data_ptr = leaf_data.append(self._data)
    
  def merge(self, data, data_leaf_coord, start_idx, end_idx, depth, leaf_data):
    if depth == -1 or (np.array_equal(data_leaf_coord[start_idx], self._data_leaf_coord) and np.array_equal(self._data_leaf_coord, data_leaf_coord[end_idx-1])):
      self._data = (self._data+np.mean(data[start_idx:end_idx], axis=0)[np.newaxis,:])/2.
      leaf_data.update(self._data, self._data_ptr)
      return self
    else:
      return OctreeNode.fromLeaf(self, depth).merge(data, data_leaf_coord, start_idx, end_idx, depth, leaf_data)

#===============================================================================
# A node, leads down the tree
#===============================================================================
class OctreeNode:
  def __init__(self):
    self._children = [OctreeEmpty(),]*8
    
  @classmethod
  def fromData(cls, data, data_leaf_coord, start_idx, end_idx, depth, leaf_data):
    return cls().merge(data, data_leaf_coord, start_idx, end_idx, depth, leaf_data)
      
  @classmethod
  def fromLeaf(cls, leaf_node, depth):
    node = cls()
    node._children[leaf_node._data_leaf_coord[depth]] = leaf_node
    return node
    
  def merge(self, data, data_leaf_coord, start_idx, end_idx, depth, leaf_data):
    counts = np.bincount(data_leaf_coord[start_idx:end_idx, depth], minlength=8)
    s_idx = 0
    e_idx = start_idx
    for child in xrange(7):
      if counts[child]:
        s_idx = e_idx
        e_idx += counts[child]
        self._children[child] = self._children[child].merge(data, data_leaf_coord, s_idx, e_idx, depth-1, leaf_data)
    return self

#===============================================================================
# The tree, saves the root node
#===============================================================================
class Octree:
  def __init__(self, data, bound_min, bound_max, size_min):
    # detect out of bound and increase bounds
    self._bound_min = bound_min
    self._bound_max = bound_max
    self._size_min = size_min
    self._max_depth = np.max(np.ceil(np.log2((self._bound_max-self._bound_min)/self._size_min)).astype(np.int64))
    
    self.leaf_data = ContiData(np.empty([1, data.shape[1]], dtype=np.float32))

    if data.shape[0] > 0:
      leaf_coord = self._getLeafCoord(data, self._max_depth)
      idx = np.lexsort(np.transpose(leaf_coord))
      self.root = Octree._makeNewNode(data[idx], leaf_coord[idx], 0, data.shape[0], self._max_depth-1, self.leaf_data)
    else:
      self.root = OctreeEmpty()
    print "Octree init done."

  def _getLeafCoord(self, data, max_depth):
    dtype_byte_size = int(np.ceil(np.log2(max_depth)/3.))
    dtype_bit_size = dtype_byte_size<<3
    return np.packbits(np.reshape(np.swapaxes(np.concatenate([np.zeros([data.shape[0], 5, max_depth], dtype=np.uint8), np.unpackbits(np.reshape(np.floor((data[:,:3]-self._bound_min)/self._size_min).astype(np.dtype('>u'+str(dtype_byte_size))).view(dtype=np.uint8), [-1,3,dtype_byte_size]), axis=2)[:,:,dtype_bit_size:dtype_bit_size-max_depth-1:-1]], axis=1), 1, 2), [data.shape[0],-1]), axis=1)
  
  def insert(self, data):
    if data.shape[0] > 0:
      leaf_coord = self._getLeafCoord(data, self._max_depth)
      idx = np.lexsort(np.transpose(leaf_coord))
      self.root.merge(data[idx], leaf_coord[idx], 0, data.shape[0], self._max_depth-1, self.leaf_data)
    
  @staticmethod
  def _makeNewNode(data, data_leaf_coord, start_idx, end_idx, depth, leaf_data):
    if depth == -1 or np.array_equal(data_leaf_coord[start_idx], data_leaf_coord[end_idx-1]):
      return OctreeLeaf(data, data_leaf_coord, start_idx, end_idx, leaf_data)
    else:
      return OctreeNode.fromData(data, data_leaf_coord, start_idx, end_idx, depth, leaf_data)

#===============================================================================
# Contiguous data memory
#===============================================================================
class ContiData:
  def __init__(self, empty):
    self._data = np.empty_like(empty)
    self._len = 0
    
  def update(self, data, idx):
    self._data[idx:idx+data.shape[0]] = data

  def append(self, data):
    if self._data.shape[0]-self._len < data.shape[0]:
      length = self._data.shape[0]
      while length-self._len < data.shape[0]:
        length *= 2
      self._data = np.resize(self._data, [length, self._data.shape[1]])
#      print self._data.shape[0]
    self._data[self._len:self._len+data.shape[0]] = data
    self._len += data.shape[0]
    return self._len-data.shape[0]
  
  def get_data(self):
    return self._data[:self._len]
  
if __name__ == '__main__':
  from time import clock
  import timeit
  data = np.repeat(np.linspace(0.,1.,300000,endpoint=False)[:,np.newaxis], 3, 1)
#  data = np.random.random([10000,8])
  t_i = clock()
  oc = Octree(data, np.array([0.,0.,0.]), np.array([1.,1.,1.]), 0.001)
  print clock()-t_i
##  print oc.leaf_data.get_data()
#  t_i = clock()
#  data = np.random.random([10000,8])
#  oc.insert(data)
#  print clock()-t_i
##  print oc.leaf_data.get_data()
  