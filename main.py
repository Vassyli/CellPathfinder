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
import argparse
import numpy as np

from contour import Contour, AlternativeContour, NewContour, NoConvergenceError, OutOfBoundaryError
from filehandler import DatastorageFileHandler
from imagefilehandler import ImageFileHandler
from fft import FFT
from barplotanimation import BarplotAnimation
from fluctuationspectra import FluctuationSpectra
from startpointdetector import StartpointDetector

#
# CONFIGURATION
#
#FILE_PREFIX = "cell-isotohyper"
#FILE_SUFFIX = ".tif"
#FILE_PATH   = "./example-video"
FILE_PREFIX = "transition"
#FILE_PREFIX = "iso_singlecell_F"
#FILE_PREFIX = "hyp3_singlecell_D"
FILE_SUFFIX = ".tif"
FILE_PATH   = ".\\example-video\\05-08-dev1-hyp1"
FILE_SUFFIX_CONTOUR = ".contour"
FILE_SUFFIX_FFT     = ".fft"
#XCUT = (91, 91+266)
#YCUT = (646, 646+252)
X_CUT = (530, 530+262)
Y_CUT = (533, 533+276)
#X_CUT = (00, 280)
#Y_CUT = (950, 1200)
STARTSEARCH_Y = 100 #180

# Settings for Startpoint Detection
X_TRANS_CUT = (0, 800)
Y_TRANS_CUT = (0, 200)

LIMIT = 100000
OFFSET = 0

#LIMIT = 1
#OFFSET = 4114 #4117

#LIMIT = 200
#OFFSET = 4000

Y_SCALE = 100
X_SCALE_START = 1
NUM_OF_BINS = 37

BAR_WIDTH = 1.0
BAR_COLOR = 'b'

##WEIGHT_MATRIX = np.array([
##    [36, 30, 24, 18, 12,  6, 12, 18, 24, 30, 36],
##    [30, 25, 20, 15, 10,  5, 10, 15, 20, 25, 30],
##    [24, 20, 16, 12,  8,  4,  8, 12, 16, 20, 24],
##    [18, 15, 12,  9,  6,  3,  6,  9, 12, 15, 18],
##    [12, 10,  8,  6,  4,  2,  4,  6,  8, 10, 12],
##    [ 6,  5,  4,  3,  2,  1,  2,  3,  4,  5,  6],
##    [12, 10,  8,  6,  4,  2,  4,  6,  8, 10, 12],
##    [18, 15, 12,  9,  6,  3,  6,  9, 12, 15, 18],
##    [24, 20, 16, 12,  8,  4,  8, 12, 16, 20, 24],
##    [30, 25, 20, 15, 10,  5, 10, 15, 20, 25, 30],
##    [36, 30, 24, 18, 12,  6, 12, 18, 24, 30, 36],
##])
WEIGHT_MATRIX = np.array([
    [9, 6, 3, 6, 9],
    [6, 4, 2, 4, 6],
    [4, 2, 1, 2, 4],
    [6, 4, 2, 4, 6],
    [9, 6, 3, 6, 9],
])
##WEIGHT_MATRIX = np.array([
##    [0, 0, 0],
##    [0, 1, 0],
##    [0, 0, 0],
##])
#WEIGHT_MATRIX = 1/WEIGHT_MATRIX
##WEIGHT_MATRIX = np.array([
##    [16, 12,  8, 4,  8, 12, 16],
##    [12,  9,  6, 3,  6,  9, 12],
##    [ 8,  6,  4, 2,  4,  1,  1],
##    [ 4,  3,  2, 1,  2,  3,  4],
##    [ 8,  6,  4, 2,  4,  1,  1],
##    [12,  9,  6, 3,  6,  9, 12],
##    [16, 12,  8, 4,  8, 12, 16],
##])

#
# DO NOT MODIFY DOWN HERE
#
def getFilename(path, pre, suf):
    return os.path.join(path, pre+suf)

contourFilename = getFilename(FILE_PATH, FILE_PREFIX, FILE_SUFFIX_CONTOUR)

def findTransition():
    fh = ImageFileHandler(FILE_PATH, FILE_PREFIX, FILE_SUFFIX)
    fh.set_limit(LIMIT, OFFSET)
    fh.cut(X_TRANS_CUT, Y_TRANS_CUT)

    trans = StartpointDetector()

    for framenumber, frame in fh:
        if framenumber%100 == 0:
            print("Frame Number", framenumber)
        trans.append(frame, sourcetype = trans.fromPIL)

    trans.plot()
    trans.show()

def createContours():
    fh = ImageFileHandler(FILE_PATH, FILE_PREFIX, FILE_SUFFIX)
    fh.set_limit(LIMIT, OFFSET)
    if len(X_CUT) == 2 and len(Y_CUT) == 2:
        fh.cut(X_CUT, Y_CUT)

    contourStorage = DatastorageFileHandler.new(contourFilename)

    for framenumber, frame in fh:
        if framenumber%100 == 0:
            print("Frame Number ", framenumber)
        #c = Contour.fromPIL(frame)
        #c = AlternativeContour.fromPIL(frame)
        #c = Contour.fromPIL(frame)
        c = NewContour.fromPIL(frame)
        #c.setWeight(WEIGHT_MATRIX)
        try:
            #c.applyDoubleSobel()
            c.findContour(yAxis = STARTSEARCH_Y)
            con = c.getContour(True)
            if len(con) > 0:
                contourStorage.append(con)
        except NoConvergenceError as e:
            print("[E] Skipped frame (NC)", framenumber)
            continue
        except OutOfBoundaryError as e:
            print("[E] Skipped frame (OoB)", framenumber)
            continue

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
    parser = argparse.ArgumentParser(description='Run some CellPathfinder methods')

    parser.add_argument('--method', nargs=1, required=True, help='Which method you\'d like to call (1, 2 or 3)')

    args = parser.parse_args()

    if '1' in args.method:
        print("Create Contour")
        createContours()
    elif '2' in args.method:
        createFFT()
    elif '3' in args.method:
        #createFluctuationSpectra()
        pass
    elif '4' in args.method:
        findTransition()
    else:
        raise ValueError("--method should only be 1, 2 or 3")

    #createContours()
    #createFFT()
    #createFluctuationSpectra()
    # createWhatever
