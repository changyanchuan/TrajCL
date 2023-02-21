# EDwP python implementation
# Paper: Indexing and Matching Trajectories under Inconsistent Sampling Rates

# This module is implemented by following the original author provided java implementation [1].
# I have tried two different python implementations [2,3] on github. 
# Neither has the same outputs as the authorized java implementation with the same inputs.
# Thus, I implemented this. -- yc
# [1]. https://www.cse.iitd.ac.in/~sayan/publications.html
# [2]. https://github.com/Rena7ssance/search-trajectory-kBCT/blob/master/distance/edwp.py
# [3]. https://github.com/skairunner/TrajectoryInspector/blob/master/EDwP/edwp.py


import sys
import math
import numpy as np
from itertools import tee

F64MAX = sys.float_info.max

class Matrix:
    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.value = np.zeros([nrow, ncol])
        self.delta = np.zeros([nrow, ncol])
        self.col_edits = np.zeros([nrow, ncol, 2])
        self.row_edits = np.zeros([nrow, ncol, 2])

        self.value[0 , 1:] = F64MAX
        self.value[1: , 0] = F64MAX

    def add(self, i, j, val, col_edit, row_edit):
        self.value[i, j] = val
        self.col_edits[i, j] = col_edit
        self.row_edits[i, j] = row_edit

    def score(self):
        return self.value[-1, -1]


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _distance(p, q):
    # p, q: list 
    return math.sqrt( (p[0] - q[0])**2 + (p[1] - q[1])**2 )


def _line_map(p1, p2, p):
    l2 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    if l2 == 0:
        return p
    t = ((p[0] - p1[0]) * (p2[0] - p1[0]) + (p[1] - p1[1]) * (p2[1] - p1[1])) / l2
    if t < 0:
        return p1
    elif t > 1:
        return p2
    else:
        return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]


def edwp(t1, t2):
    # t1, t2 : list of 2D list = [[x,y], [], ...]

    t1_len = len(t1)
    t2_len = len(t2)

    t1_edge_length = [_distance(p1, p2) for p1, p2 in _pairwise(t1)]
    t2_edge_length = [_distance(p1, p2) for p1, p2 in _pairwise(t2)]

    matrix = Matrix(t1_len, t2_len)

    total_length = sum(t1_edge_length) + sum(t2_edge_length)

    for i in range(1, matrix.nrow):
        for j in range(1, matrix.ncol):
            row_delta = col_delta = F64MAX
            row_converge1 = row_converge2 = col_converge1 = col_converge2 = F64MAX
            row_spatial_score = col_spatial_score = F64MAX
            
            t1_insert = t2_insert = None
            t1_edit = t2_edit = None

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


import time
if __name__ == '__main__':
    _time = time.time()
    t1 = [(48038,420644),(48057,420665),(48140,420818),(48196,420800),(48201,420798),(48210,420794),(48328,420790),(48335,420787),(48336,420788)]
    t2 = [(47376,420478),(48126,420826),(48140,420816),(48191,420756),(48215,420747),(48283,420761),(48351,420782),(48351,420787),(48342,420788)]
    _time = time.time()
    
    for _ in range(10000):
        rtn = edwp(t1, t2)
    print(time.time() - _time)
