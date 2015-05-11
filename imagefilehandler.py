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
from PIL import Image, ImageSequence

class ImageFileHandler:
    """ A class for handling multipart-multipage tiff files """
    path = None
    prefix = None
    suffix = None
    limit = 1000
    offset = 0
    xcut = None
    ycut = None

    list_of_files = []
    framecount = None

    def __init__(self, path, prefix, suffix):
        """
        path: This directory (and only this directory) gets searched and is used to store output files
        prefix: A common prefix among the files that have to be matched
        suffix: A common suffix among the files that have to be matched (for example, the file extension)
        """
        self.path = path
        self.prefix = prefix
        self.suffix = suffix

        # Get a List of all files that belong together
        self.list_of_files = self.searchFiles()

    def __iter__(self):
        """ Walks through every frame. """
        # Count the frames
        count = 0
        for file in self.list_of_files:
            a = Image.open(file)
            # Iteratre through the frames
            for frame in ImageSequence.Iterator(a):
                if count >= self.offset and count < (self.limit+self.offset):
                    if self.xcut == None or self.ycut == None:
                        yield count, frame
                    else:
                        yield count, frame.crop((self.xcut[0], self.ycut[0], self.xcut[1], self.ycut[1]))
                count+=1
        self.framecount = count

    def set_limit(self, limit, offset):
        """ Sets a frame-limit/offet
        lim: Max number of frames that are yielded
        offset: Yielding starts after this frame
        """
        self.limit = limit
        self.offset = offset

    def searchFiles(self):
        """ walks through the directory and looks for all files that are starting
          with self.prefix and ending with self.suffix"""
        list_of_files = []
        for dirname, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if filename.startswith(self.prefix) and filename.endswith(self.suffix):
                    list_of_files.append(os.path.join(self.path, filename))
        return list_of_files

    def getFramecount(self):
        """ returns the amount of frames that were walked through. This method is not
          callable before this class was iterated! """
        if self.framecount is None:
            raise FileHandlerException("Framecount is only available after the first iteration!")

        return self.framecount

    def getFilecount(self):
        return len(self.list_of_files)

    def cut(self, xcut, ycut):
        self.xcut = xcut
        self.ycut = ycut