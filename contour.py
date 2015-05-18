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

import traceback
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import feature

DEFAULT_WEIGHT_MATRIX = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

DEFAULT_ALTERNATIVE_WEIGHT_MATRIX = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
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

    def calculateDistance(self, p1, p2):
        return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def calculateCentroid(self):
        a = (0,0)
        N = len(self.contourPath)
        L = 0
        for i in range(0, N):
            pm = self.contourPath[i-1]
            p0 = self.contourPath[i]
            try:
                pp = self.contourPath[i+1]
            except IndexError:
                pp = self.contourPath[0]
            dm = self.calculateDistance(pm, p0)
            di = self.calculateDistance(p0, pp)
            x = a[0] + p0[0]*(dm)
            y = a[1] + p0[1]*(di)
            #x = a[0] + p0[0]
            #y = a[1] + p0[1]
            a = (x, y)
            L = L + di

        a = np.array(a)
        #c = a/(len(self.contourPath))
        c = a*1/(L) # 2*L is way too much, L is correct
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
        self.imageMatrix = abs(nd.filters.sobel(a,1)) + abs(nd.filters.sobel(a,0)) + abs(nd.filters.sobel(a,-1)) + abs(nd.filters.sobel(a, -2))
        self.imageMatrix*=-1
        self.imageMatrix+=self.imageMatrix.min()
        #a = nd.rotate(a, 0, mode='constant')
        #self.imageMatrix = feature.canny(a, sigma=3)

    def findContour(self, yAxis = None, untilX = None, maxsteps = 2000, minsteps = 100, checkPoints = 15):
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
            #print(x, black, self.imageMatrix[yAxis,x])

            if black >= min_black[1]:
                continue

            min_black = (x, black)
        #print(min_black)

        contourPath = []
        contourPath.append((min_black[0], yAxis))
        nanana = False

        # Search the contour
        i = 0
        try:
            while(True):
                nextPoint = self.findNextPoint(contourPath, checkPoints)
                if i > maxsteps:
                    raise NoConvergenceError("Maxsteps reached (%i)" % (maxsteps,))
                if i > minsteps:
                    # Minimum steps reached => check if we have reached the first point!
                    if abs(contourPath[0][0] - nextPoint[0]) <= 1 and abs(contourPath[0][1] - nextPoint[1]) <= 1:
                        contourPath.append(nextPoint)
                        break
                    if nextPoint in contourPath:
                        newPath = []
                        app = False
                        for p in contourPath:
                            if app == True:
                                newPath.append(p)
                            else:
                                if p == nextPoint:
                                    app = True
                        newPath.append(nextPoint)
                        contourPath = newPath
                        break
                contourPath.append(nextPoint)
                i+=1
                r = None
        except OutOfBoundaryError:
            print("Out of Boundary")
            r = OutOfBoundaryError
        except NoConvergenceError:
            print("No Convergence")
            r = OutOfBoundaryError

        self.contourPath = contourPath

        if r != None:
            raise r

    def findNextPoint(self, contourPath, checkPoints = 15):
        # Get newest point
        xi, yi = contourPath[-1]

        # Get points which have to be searched in order to determine the direction
        directionMask = self.getDirectionMask(contourPath, checkPoints)

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


        #print(contourPath[-1], blacksearch_intensities, minblack, directionMask)

        # Return next point
        return nextPoint

    def getDirectionMask(self, contourPath, checkPoints = 15):
        # Calculate direction or use default direction
        if len(contourPath) >= checkPoints:
            # Get the last 10 points
            slopseq = contourPath[-checkPoints:]
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

        subMatrix = (self.imageMatrix[y-weight_y_dim:y+weight_y_dim+1,x-weight_x_dim:x+weight_x_dim+1]*self.weight)/self.weightcount
        r = subMatrix.sum()
        return r

class NoConvergenceError(Exception):
    pass

class OutOfBoundaryError(Exception):
    pass

class AlternativeContour:
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
        self.setWeight(DEFAULT_ALTERNATIVE_WEIGHT_MATRIX)

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

    def calculateDistance(self, p1, p2):
        return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def calculateCentroid(self):
        a = (0,0)
        N = len(self.contourPath)
        if N == 0:
            return a
        L = 0
        for i in range(0, N):
            pm = self.contourPath[i-1]
            p0 = self.contourPath[i]
            try:
                pp = self.contourPath[i+1]
            except IndexError:
                pp = self.contourPath[0]
            dm = self.calculateDistance(pm, p0)
            di = self.calculateDistance(p0, pp)
            x = a[0] + p0[0]*(dm)
            y = a[1] + p0[1]*(di)
            a = (x, y)
            L = L + di

        a = np.array(a)
        #c = a/(len(self.contourPath))
        c = a*1/(L) # 2*L is way too much, L is correct
        return (int(round(c[0])), int(round(c[1])))

    def setMatrix(self, imageMatrix):
        self.imageMatrix = imageMatrix
        self.height, self.width = self.imageMatrix.shape

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
        c = AlternativeContour(np.array(frame))
        return c

    fromPIL = staticmethod(fromPIL)

    def applyDoubleSobel(self):
        """ Applies the sobel transformation in y and x direction and takes the absolute
        value. Then it multiplies the result with *-1 to make the biggest gradients
        the darkest colour in order to work with the blackness-contour-detection """
        a = self.imageMatrix
        self.imageMatrix = abs(nd.filters.sobel(a,1)) + abs(nd.filters.sobel(a,0)) + abs(nd.filters.sobel(a,-1)) + abs(nd.filters.sobel(a, -2))
        self.imageMatrix*=-1
        self.imageMatrix+=self.imageMatrix.min()

    def shrinkImage(self, reduction):
        new_width = self.width//reduction + (1 if self.width%reduction > 0 else 0)
        new_height = self.height//reduction + (1 if self.height%reduction > 0 else 0)
        b = np.zeros((new_height, new_width), dtype = self.imageMatrix.dtype)
        for y in range(0, self.height, reduction):
            for x in range(0, self.width, 4):
                s = sum(self.imageMatrix[y:y+reduction,x:x+reduction]/(reduction**2))
                if isinstance(s, int) == False:
                    s = sum(s)
                b[y//reduction,x//reduction] = s
        return Contour(b)

    def findContour(self, yAxis = None, untilX = None, maxsteps = 2000, minsteps = 100, reduction = 4, checkSubPoints=10):
        if yAxis == None:
            yAxis = self.height//2
        if untilX == None:
            untilX = self.width//2

        # Get small ("upper") contour
        shrunk = self.shrinkImage(reduction)
        shrunk.setWeight(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        try:
            shrunk.findContour(yAxis//reduction, untilX//reduction, maxsteps//reduction, minsteps//reduction, checkSubPoints)
            upperContour = shrunk.getContour(False)

            realContour = []

            for i in range(0, len(upperContour)):
                # Transform to "real" coordinates
                p_i = (upperContour[i][0]*reduction, upperContour[i][1]*reduction)
                p_m = (upperContour[i-1][0]*reduction, upperContour[i-1][1]*reduction) # "p minus"; point i-1
                try:
                    p_p = (upperContour[i+1][0]*reduction, upperContour[i+1][1]*reduction) # "p plus"; point i+1
                except IndexError:
                    p_p = (upperContour[0][0]*reduction, upperContour[0][1]*reduction) # point i+1 == i=0, since we have a cirlce

                # Check direction [[7, 0, 1], [6, i, 2], [5, 4, 3]]
                from_direction = self.getDirection(p_i, p_m)
                next_direction = self.getDirection(p_i, p_p)

                # Get connectivity type - corner (1) or edge (0)
                #   corners are (1, 3, 5 or 7)%2 = 1
                #   edges are (0, 2, 4 or 6)%2 = 0
                from_type = from_direction%2
                next_type = next_direction%2

                # Look for the first point and the last point
                from_points = []
                next_points = []
                maxshift = reduction-1

                if from_type == 1:
                    if from_direction == 7:
                        nw = (p_i[0], p_i[1])
                    elif from_direction == 1:
                        nw = (p_i[0]+maxshift, p_i[1])
                    elif from_direction == 3:
                        nw = (p_i[0]+maxshift, p_i[1]+maxshift)
                    else:
                        nw = (p_i[0], p_i[1]+maxshift)
                    from_points.append(nw)
                else:
                    if len(realContour) > 0:
                        last_point = realContour[-1]
                        last_point_diff = (last_point[0] - p_m[0], last_point[1] - p_m[1])
                        search_points = []
                        if from_direction == 0:
                            search_points = [
                                (last_point[0]-1, last_point[1]+1),
                                (last_point[0], last_point[1]+1),
                                (last_point[0]+1, last_point[1]+1),
                            ]
                            if last_point_diff[0] == 0:
                                search_points = search_points[1:]
                            elif last_point_diff[0] == maxshift:
                                search_points = search_points[0:-1]
                        elif from_direction == 2:
                            search_points = [
                                (last_point[0]-1, last_point[1]-1),
                                (last_point[0]-1, last_point[1]),
                                (last_point[0]-1, last_point[1]+1),
                            ]
                            if last_point_diff[1] == 0:
                                search_points = search_points[1:]
                            elif last_point_diff[1] == maxshift:
                                search_points = search_points[0:-1]
                        elif from_direction == 4:
                            search_points = [
                                (last_point[0]-1, last_point[1]-1),
                                (last_point[0], last_point[1]-1),
                                (last_point[0]+1, last_point[1]-1),
                            ]
                            if last_point_diff[0] == 0:
                                search_points = search_points[1:]
                            elif last_point_diff[0] == maxshift:
                                search_points = search_points[0:-1]
                        elif from_direction == 6:
                            search_points = [
                                (last_point[0]+1, last_point[1]-1),
                                (last_point[0]+1, last_point[1]),
                                (last_point[0]+1, last_point[1]+1),
                            ]
                            if last_point_diff[1] == 0:
                                search_points = search_points[1:]
                            elif last_point_diff[1] == maxshift:
                                search_points = search_points[0:-1]
                    # First point - needs special treatment if the last point is on an edge and not a corner.
                    else:
                        search_points = []
                        for n in range(0, reduction):
                            if from_direction == 0:
                                search_points.append((p_i[0]+n, p_i[1]))
                            elif from_direction == 2:
                                search_points.append((p_i[0]+maxshift, p_i[1]+n))
                            elif from_direction == 4:
                                search_points.append((p_i[0]+n, p_i[1]+maxshift))
                            elif from_direction == 6:
                                search_points.append((p_i[0], p_i[1]+n))
                    # Ok, we have potential first points - blackness-check
                    blackest = None
                    for n in search_points:
                        blackness = self.getBlackness(n[0], n[1])
                        if blackest == None:
                            blackest = (n, blackness)
                        else:
                            if blackness < blackest[1]:
                                blackest = (n, blackness)
                    # Add blackest point
                    from_points.append(blackest[0])

                # If possible, predict the last point
                #   (only possible if p_p and p_i are at corners)
                if next_type == 1:
                    if next_direction == 7:
                        nw = (p_i[0], p_i[1])
                    elif next_direction == 1:
                        nw = (p_i[0]+maxshift, p_i[1])
                    elif next_direction == 3:
                        nw = (p_i[0]+maxshift, p_i[1]+maxshift)
                    else:
                        nw = (p_i[0], p_i[1]+maxshift)
                    next_points.append(nw)
                else:
                    search_points = []
                    for n in range(0, reduction):
                        if next_direction == 0:
                            search_points.append((p_i[0]+n, p_i[1]))
                        elif next_direction == 2:
                            search_points.append((p_i[0]+maxshift, p_i[1]+n))
                        elif next_direction == 4:
                            search_points.append((p_i[0]+n, p_i[1]+maxshift))
                        elif next_direction == 6:
                            search_points.append((p_i[0], p_i[1]+n))
                    # Ok, we have potential first points - blackness-check
                    blackest = None
                    for n in search_points:
                        blackness = self.getBlackness(n[0], n[1])
                        if blackest == None:
                            blackest = (n, blackness)
                        else:
                            if blackness < blackest[1]:
                                blackest = (n, blackness)
                    # Add blackest point
                    next_points.append(blackest[0])

                def getSearchMatrix(point):
                    return [
                        [0, (point[0], point[1]-1), 0],
                        [0, (point[0]+1, point[1]-1), 0],
                        [0, (point[0]-1, point[1]), 0],
                        [0, (point[0]+1, point[1]), 0],
                        [0, (point[0]-1, point[1]+1), 0],
                        [0, (point[0], point[1]+1), 0],
                        [0, (point[0]+1, point[1]+1), 0],
                        [0, (point[0]-1, point[1]-1), 0],
                    ]

                # Look for new points!
                j = 0
                j_max = reduction**2
                while True:
                    if j > j_max:
                        # Ehm, yeah. No convergence - shouldn't happen, but it can.
                        # print("No Subpoint convergence, ", j)
                        break
                    if abs(next_points[-1][0] - from_points[-1][0]) <= 1 and abs(next_points[-1][1] - from_points[-1][1]) <= 1:
                        break
                    # Get all points around the last one
                    searchMatrix = getSearchMatrix(from_points[-1])
                    # Cut all points that are not in the current cell and
                    #  get the blackness of the cell
                    for s in range(0, 8):
                        drop = 0
                        # Check if x-axis is out-of-cell
                        if searchMatrix[s][1][0] < p_i[0] or searchMatrix[s][1][0] > p_i[0]+maxshift:
                            drop = 100
                        # Check if y-axis is out-of-cell
                        if searchMatrix[s][1][1] < p_i[1] or searchMatrix[s][1][1] > p_i[1]+maxshift:
                            drop = 100
                        # Check if point is already a found point
                        if searchMatrix[s][1] in from_points:
                            drop = 10
                        # Check if one of the points is the last point
                        if searchMatrix[s][1] in next_points:
                            drop = 10
                        # Direction-Specific restrictions
                        if from_direction == 0 and searchMatrix[s][1][1] == p_i[1]:
                            drop = 10
                        elif from_direction == 2 and searchMatrix[s][1][0] == (p_i[0]+maxshift):
                            drop = 10
                        elif from_direction == 4 and searchMatrix[s][1][1] == (p_i[1]+maxshift):
                            drop = 10
                        elif from_direction == 6 and searchMatrix[s][1][0] == (p_i[0]):
                            drop = 10
                        # Check if point is enclosed by found points (> n matches?)
                        subSearchMatrix = getSearchMatrix(searchMatrix[s][1])
                        subSearchCount = 0
                        n_max_matches = 2
                        for subP in subSearchMatrix:
                            if subP[1] in from_points or subP[1] in realContour[-reduction*2:]:
                                subSearchCount+=1
                        if subSearchCount >= n_max_matches:
                            drop = (subSearchCount - n_max_matches) + 1
                        # Get the blackness if not dropped
                        searchMatrix[s][0] = self.getBlackness(searchMatrix[s][1][0], searchMatrix[s][1][1])
                        searchMatrix[s][2] = drop
                    #print(searchMatrix)

                    limit = 0
                    while True:
                        clearedSearchMatrix = []
                        for p in searchMatrix:
                            if p[2] <= limit:
                                clearedSearchMatrix.append(p)
                        if len(clearedSearchMatrix) > 0:
                            break
                        limit+=1
                        if limit >= 10:
                            #print("DBG - could not find a new point, limit=", limit)
                            break
                    if len(clearedSearchMatrix) > 0:
                        from_points.append(min(clearedSearchMatrix)[1])
                    # Step counter - important, do not remove or else the while: True
                    # will run indefinitely
                    j+=1

                # ...
                dbg = np.zeros((reduction, reduction), dtype=int)

                # Add from points
                for j in range(0, len(from_points)):
                    realContour.append(from_points[j])
##                    dbg[from_points[j][1]-p_i[1],from_points[j][0]-p_i[0]] = 1
                # Add next points
                for j in range(0, len(next_points)):
                    realContour.append(next_points[-j-1])
##                    dbg[next_points[-j-1][1]-p_i[1],next_points[-j-1][0]-p_i[0]] = 1
##                for j in range(0, reduction):
##                    line = ""
##                    for k in range(0, reduction):
##                        if dbg[j,k] == 0:
##                            line += "[ ] "
##                        else:
##                            line += "[X] "
##                    print(line)
##                print("", "----------------", "")
##                #self.imageMatrix[p_i[1],p_i[0]] = self.imageMatrix.max()
            for p in realContour:
                self.imageMatrix[p[1],p[0]] = self.imageMatrix.max()
            plt.imsave("debug-doublealgo.png", self.imageMatrix)
        except Exception as e:
            print(e, type(e))
            traceback.print_exc()



        # We now have a small contour -

##        m = shrunk.imageMatrix.max()
##        plt.imsave("debug-small-blank.png", shrunk.imageMatrix)
##        for p in upperContour:
##            shrunk.imageMatrix[p[1],p[0]] = m
##        plt.imsave("debug-small.png", shrunk.imageMatrix)
        # For now
        self.contourPath = upperContour

    def getDirection(self, p1, p2):
        if p2[1] < p1[1]:
            # y_2 is smaller, means it is somewhere on top of p1
            if p2[0] < p1[0]:
                # p2 is top-left of p1
                return 7
            elif p2[0] == p1[0]:
                # p2 is top of p1
                return 0
            else:
                # p2 is top-right of p1
                return 1
        elif p2[1] > p1[1]:
            # y_2 is bigger, means it is somewhere on the bottom of p1
            if p2[0] < p1[0]:
                # p2 is bottom-left of p1
                return 5
            elif p2[0] == p1[0]:
                # p2 is bottom of p1
                return 4
            else:
                # p2 is bottom-right of p1
                return 3
        else:
            # y_2 is the same, means it is either left or right of p1
            if p2[0] < p1[0]:
                # p2 is left of p1
                return 6
            elif p2[0] == p1[0]:
                # p2 is p1
                raise ValueError("p1 and p2 have to be different points")
            else:
                # p2 is right of p1
                return 2


    def getConnectionType(self, p1, p2):
        """ Checks if p1 connects to p2 via an edge or an corner """
        if p1[0] == p2[0] or p1[1] == p2[1]:
            # Either x or y is the same => edge
            return 1 # 1 == edge
        else:
            return 0 # 0 == corner
##
##        min_black = (-1, self.imageMatrix.max())
##
##        for x in range(0, untilX):
##            try:
##                black = self.getBlackness(x, yAxis)
##            except OutOfBoundaryError:
##                black = self.imageMatrix.max()
##            #print(x, black, self.imageMatrix[yAxis,x])
##
##            if black >= min_black[1]:
##                continue
##
##            min_black = (x, black)
##        #print(min_black)
##
##        contourPath = []
##        contourPath.append((min_black[0], yAxis))
##        nanana = False
##
##        # Search the contour
##        i = 0
##        try:
##            while(True):
##                nextPoint = self.findNextPoint(contourPath)
##                if i > maxsteps:
##                    raise NoConvergenceError("Maxsteps reached (%i)" % (maxsteps,))
##                if i > minsteps:
##                    # Minimum steps reached => check if we have reached the first point!
##                    if abs(contourPath[0][0] - nextPoint[0]) <= 1 and abs(contourPath[0][1] - nextPoint[1]) <= 1:
##                        contourPath.append(nextPoint)
##                        break
##                    if nextPoint in contourPath:
##                        newPath = []
##                        app = False
##                        for p in contourPath:
##                            if app == True:
##                                newPath.append(p)
##                            else:
##                                if p == nextPoint:
##                                    app = True
##                        newPath.append(nextPoint)
##                        contourPath = newPath
##                        break
##                contourPath.append(nextPoint)
##                i+=1
##        except OutOfBoundaryError:
##            print("Out of Boundary")
##        except NoConvergenceError:
##            print("No Convergence")
##        for p in contourPath:
##            self.imageMatrix[p[1],p[0]] = self.imageMatrix.max()*5
##            #pass
##        plt.imsave("debug.png", self.imageMatrix)
##            #raise NoConvergenceError
##
##        self.contourPath = contourPath

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

    def getDirectionMask(self, contourPath, checkPoints = 15):
        # Calculate direction or use default direction
        if len(contourPath) >= checkPoints:
            # Get the last 10 points
            slopseq = contourPath[-checkPoints:]
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

        subMatrix = self.imageMatrix[y-weight_y_dim:y+weight_y_dim+1,x-weight_x_dim:x+weight_x_dim+1]/self.weightcount
        return subMatrix.sum()