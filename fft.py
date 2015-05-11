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

class FFT:
    points = []
    fft = []

    def __init__(self, points):
        self.points = points

    def getFFT(self):
        return self.fft

    def transform(self, bins = 37, includeFirst = True):
        binWidth = 360/bins
        n = len(self.points)
        # Get coordinates
        con_x = np.array([i[0] for i in self.points])
        con_y = np.array([i[1] for i in self.points])
        # Convert to polar coordinates
        con_r = np.sqrt(con_x**2 + con_y**2)
        con_p = np.arctan2(con_x, con_y) * (-180/np.pi) + 180
        con_b = con_p//binWidth
        # Prepare bin
        bin_arr = [[] for i in range(0, bins)]
        # Bin into bins
        for i in range(0, n):
            b = int(con_b[i])
            bin_arr[b].append(con_r[i])
        for i in range(0, bins):
            bin_arr[i] = sum(bin_arr[i])/len(bin_arr[i])
        # Actual Fourier-Transformation
        f = (np.fft.fft(bin_arr))
        if includeFirst:
            self.fft = [f[0].real,]
        else:
            self.fft = []
        for i in range(0, len(f)//2):
            self.fft.append(abs(((f[i+1] + f[-(i+1)]).real)))