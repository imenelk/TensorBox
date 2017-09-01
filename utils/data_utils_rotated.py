import os
import cv2
import re
import sys
import argparse
import numpy as np
import copy
import json
import annolist.AnnotationLib_rotated as al
from xml.etree import ElementTree
from scipy.misc import imread


def annotation_to_h5(H, a, cell_width, cell_height, max_len):
    region_size = H['region_size']
    assert H['region_size'] == H['image_height'] / H['grid_height']
    assert H['region_size'] == H['image_width'] / H['grid_width']
    cell_regions = get_cell_grid(cell_width, cell_height, region_size)

    cells_per_image = len(cell_regions)

    box_list = [[] for idx in range(cells_per_image)]

    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if (r.intersection(c))]

    boxes = np.zeros((1, cells_per_image, 5, max_len, 1), dtype=np.float)
    box_flags = np.zeros((1, cells_per_image, 1, max_len, 1), dtype=np.float)

    for cidx in xrange(cells_per_image):
        #assert(cur_num_boxes <= max_len)

        cell_ox, cell_oy = cell_regions[cidx].center()

        unsorted_boxes = []
        for bidx in xrange(min(len(box_list[cidx]), max_len)):

            # relative box position with respect to cell
            center_box = box_list[cidx][bidx].center()
            ox = center_box[0] - cell_ox
            oy = center_box[1] - cell_oy

            width = box_list[cidx][bidx].width()
            height = box_list[cidx][bidx].height()
            angle = box_list[cidx][bidx].angle()

            if (abs(ox) < H['focus_size'] * region_size and
                abs(oy) < H['focus_size'] * region_size and
                width < H['biggest_box_px'] and
                height < H['biggest_box_px']):
                unsorted_boxes.append(np.array([ox, oy, width, height, angle],
                                      dtype=np.float))

        for bidx, box in enumerate(sorted(unsorted_boxes,
                                   key=lambda x: x[0]**2 + x[1]**2)):
            boxes[0, cidx, :, bidx, 0] = box
            box_flags[0, cidx, 0, bidx, 0] = max(box_list[cidx][bidx].silhouetteID, 1)

    return boxes, box_flags


def get_cell_grid(cell_width, cell_height, region_size):

    cell_regions = []
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy * cell_width + ix
            ox = (ix + 0.5) * region_size
            oy = (iy + 0.5) * region_size

            r = al.AnnoRect(ox - 0.5 * region_size, oy - 0.5 * region_size,
                            ox + 0.5 * region_size, oy - 0.5 * region_size,
                            ox + 0.5 * region_size, oy + 0.5 * region_size,
                            ox - 0.5 * region_size, oy + 0.5 * region_size)
            r.track_id = cidx

            cell_regions.append(r)

    return cell_regions
