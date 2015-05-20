#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Basilius Sauter
#
# Created:     18.05.2015
# Copyright:   (c) Basilius Sauter 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os
import argparse
import numpy as np

from contour import Contour, AlternativeContour, NoConvergenceError
from filehandler import DatastorageFileHandler
from imagefilehandler import ImageFileHandler
from fft import FFT
from barplotanimation import BarplotAnimation
from fluctuationspectra import FluctuationSpectra

def createFluctuationSpectra(contourFilename):
    contourStorage = DatastorageFileHandler.load(contourFilename)
    plot = FluctuationSpectra()

    for value in contourStorage:
        plot.addVariance(value)

    plot.plot()
    contourStorage.close()
    plot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opens a given file and renders the contour fluctuations')
    parser.add_argument('contourfile')

    args = parser.parse_args()

    createFluctuationSpectra(args.contourfile)
