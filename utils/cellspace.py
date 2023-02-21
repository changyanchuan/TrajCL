import math

class CellSpace:
    def __init__(self, x_unit: int, y_unit: int, x_min, y_min, x_max, y_max):
        assert x_unit > 0 and y_unit > 0

        self.x_unit = x_unit
        self.y_unit = y_unit
        
        # whole space MBR range
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_size = int(math.ceil((x_max - x_min) / x_unit))
        self.y_size = int(math.ceil((y_max - y_min) / y_unit))


    def get_mbr(self, i_x, i_y):
        return self.x_min + self.x_unit * i_x, \
                self.y_min + self.y_unit * i_y, \
                self.x_min + self.x_unit * i_x + self.x_unit, \
                self.y_min + self.y_unit * i_y + self.y_unit


    def get_cellid_by_xyidx(self, i_x: int, i_y: int):
        return i_x * self.y_size + i_y
    
    # return (i_x, i_y)
    def get_xyidx_by_cellid(self, cell_id: int):
        return cell_id // self.y_size, cell_id % self.y_size


    def get_cellid_range(self):
        return 0, self.x_size * self.y_size - 1

    
    def size(self):
        return self.x_size * self.y_size - 1


    def get_xyidx_by_point(self, x, y):
        assert self.x_min <= x <= self.x_max \
                and self.y_min <= y <= self.y_max
        
        i_x = int(x - self.x_min) // self.x_unit
        i_y = int(y - self.y_min) // self.y_unit
        return (i_x, i_y)


    def get_cellid_by_point(self, x, y):
        i_x, i_y = self.get_xyidx_by_point(x, y)
        return self.get_cellid_by_xyidx(i_x, i_y)

    
    def neighbour_cellids(self, i_x, i_y):
        # 8 neighbours
        x_r = [i_x - 1, i_x, i_x + 1] 
        y_r = [i_y - 1, i_y, i_y + 1]
        x_r = list(filter(lambda x: 0 <= x < self.x_size, x_r))
        y_r = list(filter(lambda y: 0 <= y < self.y_size, y_r))

        xs = [l for l in x_r for _ in range(len(y_r))]
        ys = y_r * len(x_r)
        neighbours = zip(xs, ys)
        neighbours = filter(lambda xy: not (xy[0] == i_x and xy[1] == i_y), neighbours)

        return list(neighbours)


    # Added while icde revision. it could be 1 time 
    # faster than all_neighbour_cell_pairs_permutated
    def all_neighbour_cell_pairs_permutated_optmized(self):
        # (i, i) are NOT included
        # if (1, 2) in the result, no (2, 1)
        # diagonal are included
        
        all_cell_pairs = []
        all_cell_pairs_id = []
        for i_x in range(self.x_size):
            for i_y in range(1, self.y_size):
                p = ((i_x, i_y - 1), (i_x, i_y))
                all_cell_pairs.append( p )
                pid = (self.get_cellid_by_xyidx(*p[0]), self.get_cellid_by_xyidx(*p[1]))
                all_cell_pairs_id.append( pid  )

        for i_x in range(1, self.x_size):
            for i_y in range(self.y_size):
                p = ((i_x - 1, i_y), (i_x, i_y))
                all_cell_pairs.append( p )
                pid = (self.get_cellid_by_xyidx(*p[0]), self.get_cellid_by_xyidx(*p[1]))
                all_cell_pairs_id.append( pid  )

        for i_x in range(1, self.x_size):
            for i_y in range(1, self.y_size):
                p = ((i_x - 1, i_y - 1), (i_x, i_y))
                all_cell_pairs.append( p )
                pid = (self.get_cellid_by_xyidx(*p[0]), self.get_cellid_by_xyidx(*p[1]))
                all_cell_pairs_id.append( pid  )

        for i_x in range(1, self.x_size):
            for i_y in range(1, self.y_size):
                p = ((i_x - 1, i_y), (i_x, i_y - 1))
                all_cell_pairs.append( p )
                pid = (self.get_cellid_by_xyidx(*p[0]), self.get_cellid_by_xyidx(*p[1]))
                all_cell_pairs_id.append( pid  )
        
        # all_cell_pairs: [((x1, y1),(x2, y2)), ((), ()), ...]
        # all_cell_pairs_id: [(id1,id2), ( , ), ...]
        return all_cell_pairs, all_cell_pairs_id

        
    def __str__(self):
        return "unit=({},{}), xrange=({},{}), yrange=({},{}), size=({},{})".format( \
                self.x_unit, self.y_unit, self.x_min, self.x_max, self.y_min, self.y_max, \
                self.x_size, self.y_size)