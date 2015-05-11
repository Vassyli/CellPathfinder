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

import os, pickle

class DatastorageFileHandler:
    filehandler = None

    def __init__(self, filehandler):
        self.filehandler = filehandler

    def __iter__(self):
        while True:
            try:
                yield pickle.load(self.filehandler)
            except EOFError:
                break

    def new(filename):
        filehandler = open(filename, "wb")
        return DatastorageFileHandler(filehandler)

    new = staticmethod(new)

    def load(filename):
        filehandler = open(filename, "rb")
        return DatastorageFileHandler(filehandler)

    load = staticmethod(load)

    def append(self, value):
        pickle.dump(value, self.filehandler, pickle.HIGHEST_PROTOCOL)

    def close(self):
        self.filehandler.close()