"""Quaternion helpers use scalar-first (w, x, y, z) ordering."""
import numpy as np


def quat_conj(q):
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a, b):
    w, x, y, z = a
    s, u, v, t = b
    return np.asarray([w*s-x*u-y*v-z*t, w*u+x*s+y*t-z*v,
                       w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s], dtype=np.float32)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.asarray([np.cos(yaw/2), 0, 0, np.sin(yaw/2)], dtype=np.float32)


def gravity(q):
    return quat_to_matrix(q).T @ np.array([0., 0., -1.])


def tilt(q):
    return float(np.rad2deg(np.arccos(np.clip(quat_to_matrix(q)[2, 2], -1., 1.))))
