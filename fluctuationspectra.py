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

import matplotlib.pyplot as plt

from . import transform

class FluctuationSpectra:
    variance = []
    def __init__(self):
        pass

    def addVariance(self, points):
        r, p = transform.cartesian2radian(points)
        rA = 0
        rB = 0
        for i in r:
            rA = rA + i**2
            rB = rB + i
        a = rA/(len(points))
        b = rB/len(points)
        self.variance.append(a - b**2)

    def plot(self):
        plt.plot(self.variance)
        plt.ylabel("Average Radius Displacement")

    def show(self):
        plt.show()
