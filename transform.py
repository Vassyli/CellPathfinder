#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Basilius Sauter
#
# Created:     11.05.2015
# Copyright:   (c) Basilius Sauter 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def cartesian2radian(points):
    con_x = np.array([i[0] for i in points])
    con_y = np.array([i[1] for i in points])
    # Convert to polar coordinates
    con_r = np.sqrt(con_x**2 + con_y**2)
    con_p = np.arctan2(con_x, con_y) * (-180/np.pi) + 180
    # Return
    return con_r, con_p

def cart2rad(cart):
    rad = np.zeros(cart.shape, dtype=cart.dtype)
    rad[:,0] = np.sqrt(cart[:,0]**2 + cart[:,1]**2)
    rad[:,1] = np.arctan2(cart[:,0], cart[:,1]) * (-180/np.pi) + 180
    return rad

def calculateDistance(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def calculateCentroid(contour):
    a = (0,0)
    N = len(contour)
    L = 0
    for i in range(0, N):
        pm = contour[i-1]
        p0 = contour[i]
        try:
            pp = contour[i+1]
        except IndexError:
            pp = contour[0]
        dm = calculateDistance(pm, p0)
        di = calculateDistance(p0, pp)
        x = a[0] + p0[0]*(dm)
        y = a[1] + p0[1]*(di)
        a = (x, y)
        L = L + di

    a = np.array(a)
    c = a*1/(L) # 2*L is way too much, L is correct
    return c

