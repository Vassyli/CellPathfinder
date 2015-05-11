#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Basilius Sauter
#
# Created:     07.05.2015
# Copyright:   (c) Basilius Sauter 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os

from contour import Contour
from filehandler import DatastorageFileHandler
from imagefilehandler import ImageFileHandler
from fft import FFT
from barplotanimation import BarplotAnimation

#
# CONFIGURATION
#
FILE_PREFIX = "cell-isotohyper"
FILE_SUFFIX = ".tif"
FILE_PATH   = "./example-video"
FILE_SUFFIX_CONTOUR = ".contour"
FILE_SUFFIX_FFT     = ".fft"
XCUT = (91, 91+266)
YCUT = (646, 646+252)

LIMIT = 5
OFFSET = 0

Y_SCALE = 100
X_SCALE_START = 1
NUM_OF_BINS = 37

BAR_WIDTH = 1.0
BAR_COLOR = 'b'

# DO NOT MODIFY DOWN HERE
def getFilename(path, pre, suf):
    return os.path.join(path, pre+suf)

contourFilename = getFilename(FILE_PATH, FILE_PREFIX, FILE_SUFFIX_CONTOUR)

def createContours():
    fh = ImageFileHandler(FILE_PATH, FILE_PREFIX, FILE_SUFFIX)
    fh.set_limit(LIMIT, OFFSET)
    fh.cut(XCUT, YCUT)

    contourStorage = DatastorageFileHandler.new(contourFilename)

    for framenumber, frame in fh:
        c = Contour.fromPIL(frame)
        c.applyDoubleSobel()
        c.findContour()
        contourStorage.append(c.getContour(True))

    contourStorage.close()

def createFFT():
    contourStorage = DatastorageFileHandler.load(contourFilename)
    plot = BarplotAnimation(NUM_OF_BINS//2+1, BAR_WIDTH, BAR_COLOR)
    plot.set_lim((X_SCALE_START, NUM_OF_BINS//2+1), (0, Y_SCALE))

    def a(contourStorage):
        for value in contourStorage:
            fft = FFT(value)
            fft.transform(NUM_OF_BINS)
            yield fft.getFFT()

    plot.animate(a(contourStorage))
    plot.show()
    contourStorage.close()

if __name__ == '__main__':
    #createContours()
    createFFT()
    # createWhatever
