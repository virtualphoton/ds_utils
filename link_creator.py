#!/usr/bin/env python3

import os
import sys

from pathlib import Path

if len(sys.argv) not in [2, 3]:
    print("./link_creator.py <path to folder> <mask to find subfolders>")
    print("example for mask: \"homework*\"")
if len(sys.argv) == 2:
    sys.argv.append("*")

ds_utils = Path(__file__).parent
cur_dir = Path(sys.argv[1])
for subdir in cur_dir.glob(sys.argv[2]):
    print(subdir)
    os.symlink(ds_utils, subdir / "ds_utils")
