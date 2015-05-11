#-------------------------------------------------------------------------------
# Name:        CellPathfinder
# Purpose:
#
# Author:      Basilius Sauter
#
# Created:     07.05.2015
# Copyright:   (c) Basilius Sauter 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

DEFAULT_WEIGHT_MATRIX = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

class Contour:
    imageMatrix = None
    width = 0
    height = 0
    weight = None
    contourPath = []

    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3

    def __init__(self, imageMatrix):
        self.setMatrix(imageMatrix)
        self.setWeight(DEFAULT_WEIGHT_MATRIX)

    def getContour(self, centered = True):
        if centered == True:
            centroid = self.calculateCentroid()
            # Transform all coordinates
            contour = []
            for p in self.contourPath:
                contour.append((
                    p[0] - centroid[0],
                    p[1] - centroid[1]
                ))
            return contour
        else:
            return self.contourPath

    def calculateCentroid(self):
        a = (0,0)
        for p in self.contourPath:
            a = (a[0] + p[0], a[1] + p[1])
        a = np.array(a)
        c = a/len(self.contourPath)
        return (int(round(c[0])), int(round(c[1])))

    def setMatrix(self, imageMatrix):
        self.imageMatrix = imageMatrix
        self.width, self.height = self.imageMatrix.shape

    def setWeight(self, weight_matrix, weight_scale = 1):
        # Check matrix dimensions - only odd, square shapes are accepted (and only np.array type)
        if isinstance(weight_matrix, np.ndarray):
            shape = weight_matrix.shape
            if shape[0] != shape[1] or shape[0]%2 == 0:
                raise ValueError("The shape of weight_matrix needs to be uneven and squared (3x3, 5x5, etc), and not (%ix%i)" % shape)
        else:
            raise ValueError("weight_matrix has to be of type %s" % (type(np.ndarray)))
        if isinstance(weight_scale, int) == False:
            raise ValueError("weight_scale has to be of type %s" % (int))

        self.weight = weight_matrix * weight_scale
        self.weightcount = self.calcWeightcount(weight_matrix)

    def calcWeightcount(self, weight_matrix):
        c = 0
        for y in weight_matrix:
            for x in y:
                if x > 0:
                    c+=1
        return c

    def fromPIL(frame):
        frame = frame.convert("I")
        c = Contour(np.array(frame))
        return c

    fromPIL = staticmethod(fromPIL)

    def applyDoubleSobel(self):
        """ Applies the sobel transformation in y and x direction and takes the absolute
        value. Then it multiplies the result with *-1 to make the biggest gradients
        the darkest colour in order to work with the blackness-contour-detection """
        a = self.imageMatrix
        self.imageMatrix = abs(nd.filters.sobel(a,0)) + abs(nd.filters.sobel(a, 1))
        self.imageMatrix*=-1

    def findContour(self, yAxis = None, untilX = None, maxsteps = 2000, minsteps = 100):
        if yAxis == None:
            yAxis = self.height//2
        if untilX == None:
            untilX = self.width//2

        min_black = (-1, self.imageMatrix.max())

        for x in range(0, untilX):
            try:
                black = self.getBlackness(x, yAxis)
            except OutOfBoundaryError:
                black = self.imageMatrix.max()

            if black >= min_black[1]:
                continue

            min_black = (x, black)

        contourPath = []
        contourPath.append((min_black[0], yAxis))

        # Search the contour
        i = 0
        try:
            while(True):
                contourPath.append(self.findNextPoint(contourPath))
                if i > maxsteps:
                    raise NoConvergenceError("Maxsteps reached (%i)" % (maxsteps,))
                if i > minsteps:
                    # Minimum steps reached => check if we have reached the first point!
                    if abs(contourPath[0][0] - contourPath[-1][0]) <= 1 and abs(contourPath[0][1] - contourPath[-1][1]) <= 1:
                        break
                i+=1
        except OutOfBoundaryError:
            print("Out of Boundary")
        except NoConvergenceError:
            print("No Convergence")
            for p in contourPath:
                self.imageMatrix[p[1],[0]] = self.imageMatrix.max()

        self.contourPath = contourPath

    def findNextPoint(self, contourPath):
        # Get newest point
        xi, yi = contourPath[-1]

        # Get points which have to be searched in order to determine the direction
        directionMask = self.getDirectionMask(contourPath)

        # Calculate the blackness of those points
        blacksearch_intensities = []
        for point in directionMask:
            blacksearch_intensities.append(self.getBlackness(point[0], point[1]))

        # Get minimum black
        minblack = min(blacksearch_intensities)

        # Get the actual point by minium black
        nextPoint = (0, 0)
        for i in range(0, len(directionMask)):
            if blacksearch_intensities[i] == minblack:
                nextPoint = directionMask[i]

        # Return next point
        return nextPoint

    def getDirectionMask(self, contourPath, checkPoints = 10):
        # Calculate direction or use default direction
        if len(contourPath) >= checkPoints:
            # Get the last 10 points
            slopseq = contourPath[-10:]
            # Get difference in y and x direction
            y_diff = slopseq[-1][1] - slopseq[0][1]
            x_diff = slopseq[-1][0] - slopseq[0][0]
            # Get direction
            if x_diff == 0:
                if y_diff < 0:
                    direction = self.DIRECTION_UP
                else:
                    direction = self.DIRECTION_DOWN
            elif y_diff == 0:
                if x_diff > 0:
                    direction = self.DIRECTION_RIGHT
                else:
                    direction = self.DIRECTION_LEFT
            else:
                s = y_diff/x_diff
                if x_diff > 0 and y_diff < 0:
                    # s is negative
                    if s < -1:
                        direction = self.DIRECTION_UP
                    else:
                        direction = self.DIRECTION_RIGHT
                elif x_diff > 0 and y_diff > 0:
                    # s is positive
                    if s < 1:
                        direction = self.DIRECTION_RIGHT
                    else:
                        direction = self.DIRECTION_DOWN
                elif x_diff < 0 and y_diff > 0:
                    # s is negative
                    if s < -1:
                        direction = self.DIRECTION_DOWN
                    else:
                        direction = self.DIRECTION_LEFT
                else:
                    # s is positive again
                    if s < 1:
                        direction = self.DIRECTION_LEFT
                    else:
                        direction = self.DIRECTION_UP
        else:
            direction = self.DIRECTION_UP


        xi, yi = contourPath[-1]

        # Get direction mask
        if direction == self.DIRECTION_UP:
            blacksearch_points = [
                #(xi-1,yi), # left
                (xi-1,yi-1), # top-left
                (xi, yi-1), # top
                (xi+1, yi-1), # top-right
                (xi+1, yi), # right
            ]
        elif direction == self.DIRECTION_RIGHT:
            blacksearch_points = [
                #(xi, yi-1), # top
                (xi+1, yi-1), # top-right
                (xi+1, yi), # right
                (xi+1, yi+1), # bottom-right
                (xi, yi+1), # bottom
            ]
        elif direction == self.DIRECTION_DOWN:
            blacksearch_points = [
                #(xi+1, yi), # right
                (xi+1, yi+1), # bottom-right
                (xi, yi+1), # bottom
                (xi-1, yi+1), # bottom-left
                (xi-1, yi), # left
            ]
        elif direction == self.DIRECTION_LEFT:
            blacksearch_points = [
                #(xi, yi+1), # bottom
                (xi-1, yi+1), # bottom-left
                (xi-1, yi), # left
                (xi-1, yi-1), # top-left
                (xi, yi-1), # top
            ]

        return blacksearch_points

    def getBlackness(self, x, y):
        # Get weight matrix dimensions
        weight_x_dim, weight_y_dim = self.weight.shape
        # Calculate how many times we need to look "left/right" and "top/bottom"
        weight_x_dim//=2
        weight_y_dim//=2
        # Check if we have enough place to calculate the blackness (OutOfBoundaryError)
        if y < weight_y_dim or y >= (self.height - weight_y_dim - 1)  or x  < weight_x_dim or x >= (self.width - weight_x_dim - 1):
            raise OutOfBoundaryError("Out of boundary: (%i,%i)" %(x,y))

        subMatrix = self.imageMatrix[y-weight_y_dim:y+weight_y_dim+1,x-weight_x_dim:x+weight_x_dim+1]*self.weight
        return subMatrix.sum()/self.weightcount

class NoConvergenceError(Exception):
    pass

class OutOfBoundaryError(Exception):
    pass