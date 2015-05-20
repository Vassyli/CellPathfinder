#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Basilius Sauter
#
# Created:     19.05.2015
# Copyright:   (c) Basilius Sauter 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

class StartpointDetector:
    fromPIL = 1

    deviationList = []

    def __init__(self):
        pass

    def append(self, data, sourcetype = None):
        if sourcetype == self.fromPIL:
            listOfPoints = np.array(data)
            # Flatten the array
            listOfPoints.reshape(-1)
        else:
            raise AttributeError("Sourcetype cannot be None or an unsupported type")

        # Calculate standard derivation
        derv = np.std(listOfPoints)
        self.deviationList.append(derv)

    def plot(self):
        plt.plot(self.deviationList)
        plt.ylabel("Standard Deviation")

    def show(self):
        plt.show()
