# EDwP Cython implementation
# Paper: Indexing and Matching Trajectories under Inconsistent Sampling Rates

# This module is implemented by following the original author provided java implementation [1].
# I have tried two different python implementations [2,3] on github. 
# Neither has the same outputs as the authorized java implementation with the same inputs.
# Thus, I implemented this. -- yc
# [1]. https://www.cse.iitd.ac.in/~sayan/publications.html
# [2]. https://github.com/Rena7ssance/search-trajectory-kBCT/blob/master/distance/edwp.py
# [3]. https://github.com/skairunner/TrajectoryInspector/blob/master/EDwP/edwp.py

from libc.math cimport fmax
from libc.math cimport fmin
from libc.math cimport fabs
from libc.math cimport sqrt
from libc.math cimport pow
from libc.float cimport DBL_MAX

cimport numpy as np
import numpy as np

cdef double F64MAX = DBL_MAX


cdef class Matrix:
    cdef dict __dict__

    def __init__(self, int nrow, int ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.value = np.zeros([nrow, ncol])
        self.delta = np.zeros([nrow, ncol])
        self.col_edits = np.zeros([nrow, ncol, 2])
        self.row_edits = np.zeros([nrow, ncol, 2])

        self.value[0 , 1:] = F64MAX
        self.value[1: , 0] = F64MAX

    def add(self, int i, int j, double val, 
            np.ndarray[np.float64_t,ndim=1] col_edit, np.ndarray[np.float64_t,ndim=1] row_edit):
        self.value[i, j] = val
        self.col_edits[i, j] = col_edit
        self.row_edits[i, j] = row_edit

    def score(self):
        return self.value[-1, -1]


def _distance(np.ndarray[np.float64_t,ndim=1] p, np.ndarray[np.float64_t,ndim=1] q):
    return sqrt( pow((p[0] - q[0]), 2) + pow((p[1] - q[1]), 2) )


def _pairwise_distance(np.ndarray[np.float64_t,ndim=2] t):
    cdef int n = len(t)
    
    cdef np.ndarray[np.float64_t,ndim=1] dist
    dist=np.zeros((n-1))
    
    cdef int i
    for i from 1 <= i < n:
        dist[i-1] = _distance(t[i-1], t[i])
    
    return dist


def _sum_ndarray(np.ndarray[np.float64_t,ndim=1] arr):
    cdef int n = len(arr)
    cdef double l = 0.0

    cdef int i
    for i from 0 <= i < n:
        l += arr[i]
    return l


def _line_map(np.ndarray[np.float64_t,ndim=1] p1, np.ndarray[np.float64_t,ndim=1] p2, np.ndarray[np.float64_t,ndim=1] p):
    cdef double l2, t
    cdef np.ndarray[np.float64_t,ndim=1] rtn
    rtn = np.zeros((2))
    
    l2 = pow((p1[0]-p2[0]), 2) + pow((p1[1]-p2[1]), 2)
    if l2 == 0:
        return p
    t = ((p[0] - p1[0]) * (p2[0] - p1[0]) + (p[1] - p1[1]) * (p2[1] - p1[1])) / l2
    if t < 0:
        return p1
    elif t > 1:
        return p2
    else:
        rtn[0] = p1[0] + t * (p2[0] - p1[0])
        rtn[1] = p1[1] + t * (p2[1] - p1[1])
        return rtn


def edwp_c(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    # t1, t2 : list of 2D list = [[x,y], [], ...]
    cdef int t1_len, t2_len
    t1_len = len(t1)
    t2_len = len(t2)

    cdef np.ndarray[np.float64_t,ndim=1] t1_edge_length, t2_edge_length

    t1_edge_length = _pairwise_distance(t1)
    t2_edge_length = _pairwise_distance(t2)

    matrix = Matrix(t1_len, t2_len)

    cdef double total_length = _sum_ndarray(t1_edge_length) + _sum_ndarray(t2_edge_length)

    cdef double row_delta, col_delta
    cdef double row_converge1, row_converge2, col_converge1, col_converge2
    cdef double row_spatial_score, col_spatial_score
    cdef np.ndarray[np.float64_t,ndim=1] t1_insert, t2_insert, t1_edit, t2_edit
    cdef double prev_point_edge
    cdef double row_edit_distance, row_edit_edge
    cdef double col_edit_distance, col_edit_edge
    cdef double diag_coverage, sub_score, diag_score

    cdef int i, j
    for i from 1 <= i < matrix.nrow:
        for j from 1 <= j < matrix.ncol:
            row_delta = col_delta = F64MAX
            row_converge1 = row_converge2 = col_converge1 = col_converge2 = F64MAX
            row_spatial_score = col_spatial_score = F64MAX
            
            # t1_insert = t2_insert = None
            # t1_edit = t2_edit = None

            if i > 1:
                t1_edit = matrix.row_edits[i-1, j]
                t2_edit = matrix.col_edits[i-1, j]
                prev_point_edge = _distance(t1_edit, t1[i-1])
                t2_insert = _line_map(t2_edit, t2[j], t1[i-1])
                row_edit_distance = _distance(t2_insert, t1[i-1])
                row_edit_edge = _distance(t2_edit, t2_insert)
                
                row_converge1 = (row_edit_edge + prev_point_edge) / total_length
                row_converge2 = (_distance(t2[j], t2_insert) + t1_edge_length[i-1]) / total_length

                row_delta = matrix.value[i-1, j] - matrix.delta[i-1, j] + (row_edit_distance + _distance(t1_edit, t2_edit)) * row_converge1
                row_spatial_score = row_delta + (row_edit_distance + _distance(t2[j], t1[i])) * row_converge2

            if j > 1:
                t1_edit = matrix.row_edits[i, j-1]
                t2_edit = matrix.col_edits[i, j-1]
                if t1_edit is None:
                    break
                prev_point_edge = _distance(t2_edit, t2[j-1])
                t1_insert = _line_map(t1_edit, t1[i], t2[j-1])
                col_edit_distance = _distance(t1_insert, t2[j-1])
                col_edit_edge = _distance(t1_edit, t1_insert)
                
                col_converge1 = (col_edit_edge + prev_point_edge) / total_length
                col_converge2 = (_distance(t1[i], t1_insert) + t2_edge_length[j-1]) / total_length

                col_delta = matrix.value[i, j-1] - matrix.delta[i, j-1] + (col_edit_distance + _distance(t1_edit, t2_edit)) * col_converge1
                col_spatial_score = col_delta + (col_edit_distance + _distance(t1[i], t2[j])) * col_converge2

            diag_coverage = (t1_edge_length[i-1] + t2_edge_length[j-1]) / total_length
            sub_score = ( _distance(t2[j], t1[i]) + _distance(t2[j-1], t1[i-1]) ) * diag_coverage
            diag_score = matrix.value[i-1, j-1] + sub_score

            if diag_score <= col_spatial_score and diag_score <= row_spatial_score:
                matrix.add(i, j, diag_score, t2[j-1], t1[i-1])
                matrix.delta[i, j] = diag_score - matrix.value[i-1, j-1]
            elif col_spatial_score < row_spatial_score or (col_spatial_score == row_spatial_score and t2_len > t1_len):
                matrix.add(i, j, col_spatial_score, t2[j-1], t1_insert)
                matrix.delta[i, j] = col_spatial_score - col_delta
            else:
                matrix.add(i, j, row_spatial_score, t2_insert, t1[i-1])
                matrix.delta[i, j] = row_spatial_score - row_delta

    return matrix.score()
