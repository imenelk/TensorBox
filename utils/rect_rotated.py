import cv2
from shapely import geometry
import numpy as np


class Rect(object):
    def __init__(self, cx, cy, width, height, angle, confidence):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.confidence = confidence
        self.angle = angle
        self.true_confidence = confidence

    def get_coordinates(self):
        points = np.int0(cv2.boxPoints((self.cx, self.cy),
                                       (self.width, self.height),
                                       self.angle))
        return points

    def get_polygon(self):
        points = self.get_coordinates()
        return geometry.Polygon([(points[0][0], points[0][1]),
                                 (points[1][0], points[1][1]),
                                 (points[2][0], points[2][1]),
                                 (points[3][0], points[3][1])])

    def overlaps(self, other):
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()
        return self_poly.intersects(other_poly)

    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))

    def intersection(self, other):
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()
        return self_poly.intersection(other_poly).area

    def area(self):
        return self.height * self.width

    def union(self, other):
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()
        return self_poly.union(other_poly).area

    def iou(self, other):
        return self.intersection(other) / self.union(other)

    def __eq__(self, other):
        return (self.cx == other.cx and
                self.cy == other.cy and
                self.width == other.width and
                self.height == other.height and
                self.angle == other.angle and
                self.confidence == other.confidence)
