import os

from math import sqrt

import gzip
import json
import bz2
import sys
import numpy as np
import cv2
from scipy.spatial import distance as dist
import shapely
from shapely import geometry

from collections import MutableSequence

#import AnnoList_pb2
import PalLib;

import xml.dom.minidom
from xml.dom.minidom import Node
xml_dom_ext_available=False
try:
    import xml.dom.ext
    xml_dom_ext_available=True
except ImportError:
    pass


################################################
#
#  TODO: check distance function
#
################################################


def cmpAnnoRectsByScore(r1, r2):
    return cmp(r1.score, r2.score)

def cmpAnnoRectsByScoreDescending(r1, r2):
    return (-1)*cmp(r1.score, r2.score)

def cmpDetAnnoRectsByScore(r1, r2):
    return cmp(r1.rect.score, r2.rect.score);


def suffixMatch(fn1, fn2):
    l1 = len(fn1);
    l2 = len(fn2);

    if fn1[-l2:] == fn2:
        return True

    if fn2[-l1:] == fn1:
        return True

    return False

class AnnoList(MutableSequence):
    """Define a list format, which I can customize"""
    TYPE_INT32 = 5;
    TYPE_FLOAT = 2;
    TYPE_STRING = 9;

    def __init__(self, data=None):
        super(AnnoList, self).__init__()

        self.attribute_desc = {};
        self.attribute_val_to_str = {};

        if not (data is None):
            self._list = list(data)
        else:
            self._list = list()

    def add_attribute(self, name, dtype):
        _adesc = AnnoList_pb2.AttributeDesc();
        _adesc.name = name;
        if self.attribute_desc:
            _adesc.id = max((self.attribute_desc[d].id for d in self.attribute_desc)) + 1;
        else:
            _adesc.id = 0;

        if dtype == int:
            _adesc.dtype = AnnoList.TYPE_INT32;
        elif dtype == float or dtype == np.float32:
            _adesc.dtype = AnnoList.TYPE_FLOAT;
        elif dtype == str:
            _adesc.dtype = AnnoList.TYPE_STRING;
        else:
            print "unknown attribute type: ", dtype
            assert(False);

        #print "adding attribute: {}, id: {}, type: {}".format(_adesc.name, _adesc.id, _adesc.dtype);
        self.attribute_desc[name] = _adesc;

    def add_attribute_val(self, aname, vname, val):
        # add attribute before adding string corresponding to integer value
        assert(aname in self.attribute_desc);

        # check and add if new
        if all((val_desc.id != val for val_desc in self.attribute_desc[aname].val_to_str)):
            val_desc = self.attribute_desc[aname].val_to_str.add()
            val_desc.id = val;
            val_desc.s = vname;

        # also add to map for quick access
        if not aname in self.attribute_val_to_str:
            self.attribute_val_to_str[aname] = {};

        assert(not val in self.attribute_val_to_str[aname]);
        self.attribute_val_to_str[aname][val] = vname;


    def attribute_get_value_str(self, aname, val):
        if aname in self.attribute_val_to_str and val in self.attribute_val_to_str[aname]:
            return self.attribute_val_to_str[aname][val];
        else:
            return str(val);

    def save(self, fname):
        save(fname, self);

    #MA: list interface
    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if isinstance(ii, slice):
            res = AnnoList();
            res.attribute_desc = self.attribute_desc;
            res._list = self._list[ii]
            return res;
        else:
            return self._list[ii]

    def __delitem__(self, ii):
        del self._list[ii]

    def __setitem__(self, ii, val):
        self._list[ii] = val
        return self._list[ii]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """<AnnoList %s>""" % self._list

    def insert(self, ii, val):
        self._list.insert(ii, val)

    def append(self, val):
        list_idx = len(self._list)
        self.insert(list_idx, val)


def is_compatible_attr_type(protobuf_type, attr_type):
    if protobuf_type == AnnoList.TYPE_INT32:
        return (attr_type == int);
    elif protobuf_type == AnnoList.TYPE_FLOAT:
        return (attr_type == float or attr_type == np.float32);
    elif protobuf_type == AnnoList.TYPE_STRING:
        return (attr_type == str);
    else:
        assert(false);


def protobuf_type_to_python(protobuf_type):
    if protobuf_type == AnnoList.TYPE_INT32:
        return int;
    elif protobuf_type == AnnoList.TYPE_FLOAT:
        return float;
    elif protobuf_type == AnnoList.TYPE_STRING:
        return str;
    else:
        assert(false);


class AnnoPoint(object):
    def __init__(self, x=None, y=None, id=None):
        self.x = x;
        self.y = y;
        self.id = id;

class AnnoRect(object):
    def __init__(self, x1=-1, y1=-1, x2=-1, y2=-1, x3=-1, y3=-1, x4=-1, y4=-1):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4

        self.score = -1.0
        self.scale = -1.0
        self.articulations =[]
        self.viewpoints =[]
        self.d3 = []

        self.silhouetteID = -1
        self.classID = -1
        self.track_id = -1

        self.point = [];
        self.at = {};

    def get_rotated_rectangle(self):
        points = np.array([[self.x1, self.y1],
                           [self.x2, self.y2],
                           [self.x3, self.y3],
                           [self.x4, self.y4]])
        return cv2.minAreaRect(points)

    def width(self):
        _, dim, _ = self.get_rotated_rectangle(self)
        return dim[0]

    def height(self):
        _, dim, _ = self.get_rotated_rectangle(self)
        return dim[1]

    def center(self):
        center, _, _ = self.get_rotated_rectangle(self)
        return center

    def angle(self):
        _, _, angle = self.get_rotated_rectangle(self)
        return angle

    def left(self):
        return min(self.x1, self.x2)

    def right(self):
        return max(self.x1, self.x2)

    def top(self):
        return min(self.y1, self.y2)

    def bottom(self):
        return max(self.y1, self.y2)

    def forceAspectRatio(self, ratio, KeepHeight = False, KeepWidth = False):
        """force the Aspect ratio"""
        if KeepWidth or ((not KeepHeight) and self.width() * 1.0 / self.height() > ratio):
            # extend height
            newHeight = self.width() * 1.0 / ratio
            points = np.int0(cv2.boxPoints((self.center(),
                                            (self.width(), newHeight),
                                            self.angle())))
            self.x1, self.y1 = points[0]
            self.x2, self.y2 = points[1]
            self.x3, self.y3 = points[2]
            self.x4, self.y4 = points[3]

        else:
            # extend width
            newWidth = self.height() * ratio
            points = np.int0(cv2.boxPoints((self.center(),
                                            (newWidth, self.height()),
                                            self.angle())))
            self.x1, self.y1 = points[0]
            self.x2, self.y2 = points[1]
            self.x3, self.y3 = points[2]
            self.x4, self.y4 = points[3]

    def clipToImage(self, min_x, max_x, min_y, max_y):
        self.x1 = max(min_x, self.x1)
        self.x2 = max(min_x, self.x2)
        self.x3 = max(min_x, self.x3)
        self.x4 = max(min_x, self.x4)

        self.y1 = max(min_y, self.y1)
        self.y2 = max(min_y, self.y2)
        self.y3 = max(min_y, self.y3)
        self.y4 = max(min_y, self.y4)

        self.x1 = min(max_x, self.x1)
        self.x2 = min(max_x, self.x2)
        self.x3 = min(max_x, self.x3)
        self.x4 = min(max_x, self.x4)

        self.y1 = min(max_y, self.y1)
        self.y2 = min(max_y, self.y2)
        self.y3 = min(max_y, self.y3)
        self.y4 = min(max_y, self.y4)

    def printContent(self):
        print "Coords: ", self.x1, self.y1, self.x2, self.y2, self.x3, self.y3,
        self.x4, self.y4
        print "Score: ", self.score
        print "Articulations: ", self.articulations
        print "Viewpoints: ", self.viewpoints
        print "Silhouette: ", self.silhouetteID

    def writeJSON(self):
        jdoc = {"x1": self.x1, "x2": self.x2, "y1": self.y1, "y2": self.y2,
                "x3": self.x3, "x4": self.x4, "y3": self.y3, "y4": self.y4}

        if (self.score != -1):
            jdoc["score"] = self.score
        return jdoc

    def sortCoords(self):
        pts = np.array([[self.x1, self.y1],
                        [self.x2, self.y2],
                        [self.x3, self.y3],
                        [self.x4, self.y4]])
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        self.x1, self.y1 = tl
        self.x2, self.y2 = tr
        self.x3, self.y3 = br
        self.x4, self.y4 = bl

    def rescale(self, factor):
        self.x1 = (self.x1*float(factor))
        self.y1 = (self.y1*float(factor))
        self.x2 = (self.x2*float(factor))
        self.y2 = (self.y2*float(factor))
        self.x3 = (self.x3*float(factor))
        self.y3 = (self.y3*float(factor))
        self.x4 = (self.x4*float(factor))
        self.y4 = (self.y4*float(factor))

    def resize(self, factor, factor_y = None):
        w = self.width()
        h = self.height()
        if factor_y is None:
            factor_y = factor
        points = np.int0(cv2.boxPoints((self.center(),
                                        (w*factor, h*factor_y),
                                        self.angle())))
        self.x1, self.y1 = points[0]
        self.x2, self.y2 = points[1]
        self.x3, self.y3 = points[2]
        self.x4, self.y4 = points[3]

    def get_polygon(self):
        return geometry.Polygon([(self.x1, self.y1), (self.x2, self.y2),
                                 (self.x3, self.y3), (self.x4, self.y4)])

    def intersection(self, other):
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()

        return self_poly.intersection(other_poly).area

    def cover(self, other):
        nWidth = self.width()
        nHeight = self.height()
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()

        return self_poly.intersection(other_poly).area / float(nWidth * nHeight)

    def overlap_pascal(self, other):
        self_poly = self.get_polygon()
        other_poly = other.get_polygon()

        return self_poly.intersection(other_poly).area / self_poly.union(other_poly).area

    def isMatchingPascal(self, other, minOverlap):
        overlap = self.overlap_pascal(other)
        if (overlap >= minOverlap and (self.classID == -1 or other.classID == -1 or self.classID == other.classID)):
            return 1
        else:
            return 0

    def distance(self, other, aspectRatio=-1, fixWH='fixheight'):
        if (aspectRatio!=-1):
            if (fixWH=='fixwidth'):
                dWidth  = float(self.x2 - self.x1)
                dHeight = dWidth / aspectRatio
            elif (fixWH=='fixheight'):
                dHeight = float(self.y2 - self.y1)
                dWidth  = dHeight * aspectRatio
        else:
            dWidth  = float(self.x2 - self.x1)
            dHeight = float(self.y2 - self.y1)

        xdist = (self.x1 + self.x2 - other.x1 - other.x2) / dWidth
        ydist = (self.y1 + self.y2 - other.y1 - other.y2) / dHeight

        return sqrt(xdist*xdist + ydist*ydist)

    def isMatchingStd(self, other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1):
        cover = other.cover(self)
        overlap = self.cover(other)
        dist = self.distance(other, aspectRatio, fixWH)

        #if(self.width() == 24 ):
        #print cover, " ", overlap, " ", dist
        #print coverThresh, overlapThresh, distThresh
        #print (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh)

        if (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh and self.classID == other.classID):
            return 1
        else:
            return 0

    def isMatching(self, other, style, coverThresh, overlapThresh, distThresh, minOverlap, aspectRatio=-1, fixWH=-1):
        #choose matching style
        if (style == 0):
            return self.isMatchingStd(other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1)

        if (style == 1):
            return self.isMatchingPascal(other, minOverlap)



class Annotation(object):

    def __init__(self):
        self.imageName = ""
        self.imagePath = ""
        self.rects =[]
        self.frameNr = -1

    def clone_empty(self):
        new = Annotation()
        new.imageName = self.imageName
        new.imagePath = self.imagePath
        new.frameNr   = self.frameNr
        new.rects     = []
        return new

    def filename(self):
        return os.path.join(self.imagePath, self.imageName)

    def printContent(self):
        print "Name: ", self.imageName
        for rect in self.rects:
            rect.printContent()

    def writeJSON(self):
        jdoc = {}
        jdoc['image_path'] = os.path.join(self.imagePath, self.imageName)
        jdoc['rects'] = []
        for rect in self.rects:
            jdoc['rects'].append(rect.writeJSON())
        return jdoc

    def sortByScore(self, dir="ascending"):
        if (dir=="descending"):
            self.rects.sort(cmpAnnoRectsByScoreDescending)
        else:
            self.rects.sort(cmpAnnoRectsByScore)

    def __getitem__(self, index):
        return self.rects[index]

class detAnnoRect:
    def __init(self):
        self.imageName = ""
        self.frameNr = -1
        self.rect = AnnoRect()
        self.imageIndex = -1
        self.boxIndex = -1

#####################################################################
# Parsing


def parseJSON(filename):
    filename = os.path.realpath(filename)
    name, ext = os.path.splitext(filename)
    assert ext == '.json'

    annotations = AnnoList([])
    with open(filename, 'r') as f:
        jdoc = json.load(f)

    for annotation in jdoc:
        anno = Annotation()
        anno.imageName = annotation["image_path"]

        rects = []
        for annoRect in annotation["rects"]:
            rect = AnnoRect()

            rect.x1 = annoRect["x1"]
            rect.x2 = annoRect["x2"]
            rect.x3 = annoRect["x3"]
            rect.x4 = annoRect["x4"]
            rect.y1 = annoRect["y1"]
            rect.y2 = annoRect["y2"]
            rect.y3 = annoRect["y3"]
            rect.y4 = annoRect["y4"]
            if "score" in annoRect:
                rect.score = annoRect["score"]

            rects.append(rect)

        anno.rects = rects
        annotations.append(anno)

    return annotations


def parse(filename, abs_path=False):
    #print "Parsing: ", filename
    name, ext = os.path.splitext(filename)

    annolist = parseJSON(filename)

    if abs_path:
        basedir = os.path.dirname(os.path.abspath(filename))
        for a in annolist:
            a.imageName = basedir + "/" + os.path.basename(a.imageName)

    return annolist


#####################################################################
### Saving

def save(filename, annotations):
    print "saving: ", filename;
    name, ext = os.path.splitext(filename)
    return saveJSON(filename, annotations)

def saveJSON(filename, annotations):
    [name, ext] = os.path.splitext(filename)

    jdoc = []
    for annotation in annotations:
        jdoc.append(annotation.writeJSON())

    with open(filename, 'w') as f:
        f.write(json.dumps(jdoc, indent=2, sort_keys=True))

#####################################################################
### Statistics

def getStats(annotations):
    no = 0
    noTiny =0
    noSmall =0
    heights = []
    widths =[]

    ###--- get all rects ---###
    for anno in annotations:
        no = no + len(anno.rects)
        for rect in anno.rects:
            if (rect.height()<36):
                noTiny=noTiny+1
            if (rect.height()<128):
                noSmall=noSmall+1
            heights.append(rect.height())
            if (rect.width()==0):
                print "Warning: width=0 in image ", anno.imageName
                widths.append(1)
            else:
                widths.append(rect.width())
                if (float(rect.height())/float(rect.width())<1.5):
                    print "Degenerated pedestrian annotation: ", anno.imageName

    ###--- compute average height and variance ---###
    avgHeight = 0
    varHeight = 0


    minHeight = 0
    maxHeight = 0
    if len(heights) > 0:
        minHeight = heights[0]
        maxHeight = heights[0]

    for height in heights:
        avgHeight = avgHeight+height
        if (height > maxHeight):
            maxHeight = height
        if (height < minHeight):
            minHeight = height

    if (no>0):
        avgHeight = avgHeight/no
    for height in heights:
        varHeight += (height-avgHeight)*(height-avgHeight)
    if (no>1):
        varHeight=float(varHeight)/float(no-1)

    ###--- compute average width and variance ---###
    avgWidth = 0
    varWidth = 0
    for width in widths:
        avgWidth = avgWidth+width
    if (no>0):
        avgWidth = avgWidth/no
    for width in widths:
        varWidth += (width-avgWidth)*(width-avgWidth)

    if (no>1):
        varWidth=float(varWidth)/float(no-1)

    ###--- write statistics ---###
    print "  Total # rects:", no
    print "     avg. Width:", avgWidth, " (", sqrt(varWidth), "standard deviation )"
    print "    avg. Height:", avgHeight, " (", sqrt(varHeight), "standard deviation )"
    print "     tiny rects:", noTiny, " (< 36 pixels)"
    print "    small rects:", noSmall, " (< 128 pixels)"
    print "    minimum height:", minHeight
    print "    maximum height:", maxHeight

    ###--- return ---###
    return [widths, heights]

############################################################
##
##  IDL merging
##

def mergeIDL(detIDL, det2IDL, detectionFuse= True, minOverlap = 0.5):
    mergedIDL = []

    for i,anno in enumerate(detIDL):
        mergedAnno = Annotation()
        mergedAnno.imageName = anno.imageName
        mergedAnno.frameNr = anno.frameNr
        mergedAnno.rects = anno.rects

        imageFound = False
        filterIndex = -1
        for i,filterAnno in enumerate(det2IDL):
            if (suffixMatch(anno.imageName, filterAnno.imageName) and anno.frameNr == filterAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            mergedIDL.append(mergedAnno)
            continue

        for rect in det2IDL[filterIndex].rects:
            matches = False

            for frect in anno.rects:
                if rect.overlap_pascal(frect) > minOverlap:
                    matches = True
                    break

            if (not matches or detectionFuse == False):
                mergedAnno.rects.append(rect)

        mergedIDL.append(mergedAnno)

    return mergedIDL


############################################################################33
#
# Function to force the aspect ratio of annotations to ratio = width / height
#
#
def forceAspectRatio(annotations, ratio, KeepHeight = False, KeepWidth = False):
    for anno in annotations:
        for rect in anno.rects:
            rect.forceAspectRatio(ratio, KeepHeight, KeepWidth)
            #Determine which side needs to be extended
#                       if (rect.width() * 1.0 / rect.height() > ratio):
#
#                               #Too wide -> extend height
#                               newHeight = rect.width() * 1.0 / ratio
#                               rect.y1 = int(rect.centerY() - newHeight / 2.0)
#                               rect.y2 = int(rect.y1 + newHeight)
#
#                       else:
#                               #Too short -> extend width
#                               newWidth = rect.height() * ratio
#                               rect.x1 = int(rect.centerX() - newWidth / 2.0)
#                               rect.x2 = int(rect.x1 + newWidth)


###################################################################
# Function to greedyly remove subset detIDL from gtIDL
#
# returns two sets
#
# [filteredIDL, missingRecallIDL]
#
# filteredIDL == Rects that were present in both sets
# missingRecallIDL == Rects that were only present in set gtIDL
#
###################################################################
def extractSubSet(gtIDL, detIDL):
    filteredIDL = []
    missingRecallIDL = []

    for i,gtAnno in enumerate(gtIDL):
        filteredAnno = Annotation()
        filteredAnno.imageName = gtAnno.imageName
        filteredAnno.frameNr = gtAnno.frameNr

        missingRecallAnno = Annotation()
        missingRecallAnno.imageName = gtAnno.imageName
        missingRecallAnno.frameNr = gtAnno.frameNr

        imageFound = False
        filterIndex = -1
        for i,anno in enumerate(detIDL):
            if (suffixMatch(anno.imageName, gtAnno.imageName) and anno.frameNr == gtAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            print "Image not found " + gtAnno.imageName + " !"
            missingRecallIDL.append(gtAnno)
            filteredIDL.append(filteredAnno)
            continue

        matched = [-1] * len(detIDL[filterIndex].rects)
        for j, rect in enumerate(gtAnno.rects):
            matches = False

            matchingID = -1
            minCenterPointDist = -1
            for k,frect in enumerate(detIDL[filterIndex].rects):
                minCover = 0.5
                minOverlap = 0.5
                maxDist = 0.5

                if rect.isMatchingStd(frect, minCover,minOverlap, maxDist):
                    if (matchingID == -1 or rect.distance(frect) < minCenterPointDist):
                        matchingID = k
                        minCenterPointDist = rect.distance(frect)
                        matches = True

            if (matches):
                #Already matched once check if you are the better match
                if(matched[matchingID] >= 0):
                    #Take the match with the smaller center point distance
                    if(gtAnno.rects[matched[matchingID]].distance(frect) > rect.distance(frect)):
                        missingRecallAnno.rects.append(gtAnno.rects[matched[matchingID]])
                        filteredAnno.rects.remove(gtAnno.rects[matched[matchingID]])
                        filteredAnno.rects.append(rect)
                        matched[matchingID] = j
                    else:
                        missingRecallAnno.rects.append(rect)
                else:
                    #Not matched before.. go on and add the match
                    filteredAnno.rects.append(rect)
                    matched[matchingID] = j
            else:
                missingRecallAnno.rects.append(rect)

        filteredIDL.append(filteredAnno)
        missingRecallIDL.append(missingRecallAnno)

    return (filteredIDL     , missingRecallIDL)

###########################################################
#
#  Function to remove all detections with a too low score
#
#
def filterMinScore(detections, minScore):
    newDetections = []
    for anno in detections:
        newAnno = Annotation()
        newAnno.frameNr = anno.frameNr
        newAnno.imageName = anno.imageName
        newAnno.imagePath = anno.imagePath
        newAnno.rects = []

        for rect in anno.rects:
            if(rect.score >= minScore):
                newAnno.rects.append(rect)

        newDetections.append(newAnno)
    return newDetections

# foo.idl -> foo-suffix.idl, foo.idl.gz -> foo-suffix.idl.gz etc
def suffixIdlFileName(filename, suffix):
    exts = [".idl", ".idl.gz", ".idl.bz2"]
    for ext in exts:
        if filename.endswith(ext):
            return filename[0:-len(ext)] + "-" + suffix + ext
    raise ValueError("this does not seem to be a valid filename for an idl-file")

if __name__ == "__main__":
# test output
    idl = parseIDL("/tmp/asdf.idl")
    idl[0].rects[0].articulations = [4,2]
    idl[0].rects[0].viewpoints = [2,3]
    saveXML("", idl)


def annoAnalyze(detIDL):
    allRects = []

    for i,anno in enumerate(detIDL):
        for j in anno.rects:
            newRect = detAnnoRect()
            newRect.imageName = anno.imageName
            newRect.frameNr = anno.frameNr
            newRect.rect = j
            allRects.append(newRect)

    allRects.sort(cmpDetAnnoRectsByScore)

    filteredIDL = AnnoList([])
    for i in allRects:
        a = Annotation()
        a.imageName = i.imageName
        a.frameNr = i.frameNr
        a.rects = []
        a.rects.append(i.rect)
        filteredIDL.append(a)

    return filteredIDL
