import numpy as np
import random
import json
import os
import cv2
from shapely.ops import cascaded_union
import itertools
from scipy.misc import imread, imresize
import tensorflow as tf

from data_utils_rotated import annotation_to_h5
from utils.annolist import AnnotationLib_rotated as al
from rect_rotated import Rect
from utils import tf_concat


def rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        r.x1 *= x_scale
        r.x2 *= x_scale
        r.x3 *= x_scale
        r.x4 *= x_scale

        r.y1 *= y_scale
        r.y2 *= y_scale
        r.y3 *= y_scale
        r.y4 *= y_scale
    return anno


def load_idl_tf(idlfile, H, jitter):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = []
    for anno in annolist:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        annos.append(anno)
    random.seed(0)
    if H['data']['truncate_data']:
        annos = annos[:10]
    for epoch in itertools.count():
        random.shuffle(annos)
        for anno in annos:
            try:
                if H['grayscale']:
                    I = imread(anno.imageName,
                               mode='RGB'
                               if random.random() < H['grayscale_prob']
                               else 'L')

                    if len(I.shape) < 3:
                        I = cv2.cvtColor(I, cv2.COLOR_GRAY2RGB)
                else:
                    I = imread(anno.imageName, mode='RGB')
                if (I.shape[0] != H["image_height"] or
                   I.shape[1] != H["image_width"]):
                    if epoch == 0:
                        anno = rescale_boxes(I.shape, anno, H["image_height"],
                                             H["image_width"])
                    I = imresize(I, (H["image_height"], H["image_width"]),
                                 interp='cubic')

                boxes, flags = annotation_to_h5(H,
                                                anno,
                                                H["grid_width"],
                                                H["grid_height"],
                                                H["rnn_len"])

                yield {"image": I, "boxes": boxes, "flags": flags}
            except:
                print("The file is not exist {}".format(anno.imageName))


def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v


def load_data_gen(H, phase, jitter):
    grid_size = H['grid_width'] * H['grid_height']

    data = load_idl_tf(H["data"]['%s_idl' % phase], H,
                       jitter={'train': jitter, 'test': False}[phase])

    for d in data:
        output = {}

        rnn_len = H["rnn_len"]
        flags = d['flags'][0, :, 0, 0:rnn_len, 0]
        boxes = np.transpose(d['boxes'][0, :, :, 0:rnn_len, 0], (0, 2, 1))
        assert(flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 5))

        output['image'] = d['image']
        output['confs'] = np.array([[make_sparse(int(detection),
                                    d=H['num_classes']) for detection in cell]
                                    for cell in flags])
        output['boxes'] = boxes
        output['flags'] = flags

        yield output


def add_rectangles(H, orig_image, confidences, boxes, use_stitching=False,
                   rnn_len=1, min_conf=0.1, show_removed=True, tau=0.25,
                   show_suppressed=True):
    image = np.copy(orig_image[0])
    boxes_r = np.reshape(boxes, (-1,
                                 H["grid_height"],
                                 H["grid_width"],
                                 rnn_len,
                                 5))
    confidences_r = np.reshape(confidences, (-1,
                                             H["grid_height"],
                                             H["grid_width"],
                                             rnn_len,
                                             H['num_classes']))
    cell_pix_size = H['region_size']
    all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
    for n in range(rnn_len):
        for y in range(H["grid_height"]):
            for x in range(H["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                angle = bbox[4]
                conf = np.max(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect(abs_cx, abs_cy, w, h, angle, conf))

    all_rects_r = [r for row in all_rects for cell in row for r in cell]

    # TO-DO use cascaded_union (from shapely lib) to do the stitching
    # We will need to filter the rectangles before computing the union
    # WARNING : cascaded_union does not necessarily output a rectangle
    if use_stitching:
        #from stitch_wrapper import stitch_rects
        #acc_rects = stitch_rects(all_rects, tau)
        acc_rects = cascaded_union([r.get_polygon() for r in all_rects_r])
    else:
        acc_rects = all_rects_r

    if show_suppressed:
        pairs = [(all_rects_r, (255, 0, 0))]
    else:
        pairs = []
    pairs.append((acc_rects, (0, 255, 0)))
    for rect_set, color in pairs:
        for rect in rect_set:
            if rect.confidence > min_conf:
                box = np.int0(cv2.BoxPoints((rect.cx, rect.cy),
                                            (rect.width, rect.height),
                                            rect.angle))
                cv2.drawContours(image, [box], 0, color, 1)

    cv2.putText(image, str(len(filter(lambda rect: rect.confidence > min_conf,
                                      acc_rects))),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    rects = []
    for rect in acc_rects:
        r = al.AnnoRect()
        points = rect.get_coordinates()
        r.x1, r.y1 = points[0]
        r.x2, r.y2 = points[1]
        r.x3, r.y3 = points[2]
        r.x4, r.y4 = points[3]
        r.score = rect.true_confidence
        rects.append(r)

    return image, rects
