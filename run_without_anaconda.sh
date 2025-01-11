#!/bin/bash
unset LD_LIBRARY_PATH  # anaconda path
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6  # force lbstdc++
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu  # set path
yourEnv/bin/python "$@"
