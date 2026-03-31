#
# Copyright (C) 2023, Inria GRAPHDECO research group
# Copyright (C) 2025, New York University
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# Original 3D Gaussian Splatting code from:
# https://github.com/graphdeco-inria/gaussian-splatting
#
# CLM-GS modifications by NYU Systems Group
# https://github.com/nyu-systems/CLM-GS
#

from errno import EEXIST
from os import makedirs, path
import os


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)