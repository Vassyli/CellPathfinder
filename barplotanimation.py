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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BarplotAnimation:
    fig = None
    ax = None
    bars = None

    numOfBins = 0

    def __init__(self, numOfBins, barWidth, barColor):
        self.fig, self.ax = plt.subplots()
        self.numOfBins = numOfBins
        self.bars = self.ax.bar(np.arange(numOfBins), [i for i in range(numOfBins)], barWidth, color=barColor)

    def set_lim(self, xlim, ylim):
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])

    def animate(self, generator):
        ani = animation.FuncAnimation(self.fig, self._animateHelper, generator)

    def _animateHelper(self, i):
        for a in range(0, self.numOfBins):
            self.bars[a].set_height(i[a])

    def show(self):
        plt.show()