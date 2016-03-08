'''
Created on Feb 2, 2016

@author: lcc
'''
import numpy as np

def inv_rot(rot):
  R_t = np.transpose(rot[:3,:3])
  return np.vstack([np.hstack([R_t, np.dot(-R_t, rot[:3,3][:,np.newaxis])]), np.array([0., 0., 0., 1.], dtype=np.float)])

def mat_to_quat(mat):
  return mat_to_quat2(mat)

def mat_to_quat1(mat):
  qw = -np.sqrt(1.+mat[0,0]+mat[1,1]+mat[2,2])/2.
  return np.array([mat[0,3],mat[1,3],mat[2,3],(mat[2,1]-mat[1,2])/(4.*qw),(mat[0,2]-mat[2,0])/(4.*qw),(mat[1,0]-mat[0,1])/(4.*qw),qw], dtype=np.float)

def mat_to_quat2(mat):
  p_2 = np.arctan2(-mat[2,0], np.sqrt(np.square(mat[0,0])+np.square(mat[1,0])))/2.
  if p_2 >= 45:
    p_2 = 45.
    y_2 = np.arctan2(mat[1,2],mat[0,2])/2.
    r_2 = 0.
  elif p_2 <= -45.:
    p_2 = -45.
    y_2 = np.arctan2(-mat[1,2],-mat[0,2])/2.
    r_2 = 0.
  else:
    y_2 = np.arctan2(mat[1,0],mat[0,0])/2.
    r_2 = np.arctan2(mat[2,1],mat[2,2])/2.
  qx = np.sin(r_2)*np.cos(p_2)*np.cos(y_2)-np.cos(r_2)*np.sin(p_2)*np.sin(y_2)
  qy = np.cos(r_2)*np.sin(p_2)*np.cos(y_2)+np.sin(r_2)*np.cos(p_2)*np.sin(y_2)
  qz = np.cos(r_2)*np.cos(p_2)*np.sin(y_2)-np.sin(r_2)*np.sin(p_2)*np.cos(y_2)
  qw = np.cos(r_2)*np.cos(p_2)*np.cos(y_2)+np.sin(r_2)*np.sin(p_2)*np.sin(y_2)
  return np.array([mat[0,3],mat[1,3],mat[2,3],-qx,-qy,-qz,-qw], dtype=np.float)

def quat_to_mat_2(quat):
  xx = quat[3]*quat[3]
  xy = quat[3]*quat[4]
  xz = quat[3]*quat[5]
  xw = quat[3]*quat[6]
  yy = quat[4]*quat[4]
  yz = quat[4]*quat[5]
  yw = quat[4]*quat[6]
  zz = quat[5]*quat[5]
  zw = quat[5]*quat[6]
  return np.array([[1.-2.*(zz+yy),
                    2.*(xy-zw),
                    2.*(yw+xz),
                    quat[0]],
                   [2.*(xy+zw),
                    1.-2.*(zz+xx),
                    2.*(yz-xw),
                    quat[1]],
                   [2.*(xz-yw),
                    2.*(yz+xw),
                    1.-2.*(yy+xx),
                    quat[2]],
                   [0., 0., 0., 1.]], dtype=np.float)
  
def quat_to_mat_1(quat):
  xx = quat[3]*quat[3]
  xy = quat[3]*quat[4]
  xz = quat[3]*quat[5]
  xw = quat[3]*quat[6]
  yy = quat[4]*quat[4]
  yz = quat[4]*quat[5]
  yw = quat[4]*quat[6]
  zz = quat[5]*quat[5]
  zw = quat[5]*quat[6]
  ww = quat[6]*quat[6]
  return np.array([[ww+xx-yy-zz,
                    2.*(xy-zw),
                    2.*(yw+xz),
                    quat[0]],
                   [2.*(xy+zw),
                    ww-xx+yy-zz,
                    2.*(yz-xw),
                    quat[1]],
                   [2.*(xz-yw),
                    2.*(yz+xw),
                    ww-xx-yy+zz,
                    quat[2]],
                   [0., 0., 0., 1.]], dtype=np.float)
  
def quat_to_mat(quat):
  return quat_to_mat_2(quat)
  
  
