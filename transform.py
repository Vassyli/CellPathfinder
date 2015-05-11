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