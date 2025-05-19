#!/usr/bin/python3

import json
import sys
import os
import string
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

if sys.argv[1] == "no_float":
    retval = "+define+ECE411_NO_FLOAT"
    print(retval)

if sys.argv[1] == "arch":
    retval = "rv32i"
    print(retval)

if sys.argv[1] == "abi":
    retval = "ilp32"
    print(retval)
