import sys
sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def straight(src):
    return src


def simplify(src):
    # src: [[lon, lat], [lon, lat], ...]
    return rdp(src, epsilon = Config.traj_simp_dist)


def shift(src):
    return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src):
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace = False)
    return np.delete(arr, mask_idx, 0).tolist()


def subset(src):
    l = len(src)
    max_start_idx = l - int(l * Config.traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * Config.traj_subset_ratio)
    return src[start_idx: end_idx]


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'subset': subset}.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace):
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (Config.trajcl_local_mask_sidelen / 1.414) # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, dist, radian] )

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0] )
    
    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.append( [x, y, 0.0, 0.0] )
    # tgt = [length, 4]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

