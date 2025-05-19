.section .text
<<<<<<< HEAD
.global _start
_start:
    lui   x1, 0xAAAAA 
    addi  x1, x1, 2000
    nop
    nop
    nop
    nop
    nop
    lw x2, 0(x1)
    addi x2, x2, 2
    nop               # preventing fetching illegal instructions
    nop
    nop
    nop
    nop
    lui   x1, 0xAAAAA 
    nop
    addi  x1, x1, 2000
    nop
    nop
    nop
    nop
    nop
    lw x2, 0(x1)
    nop
    addi x2, x2, 2
    nop               # preventing fetching illegal instructions
    nop
    nop
    nop
    nop
=======
.globl _start
_start:
    addi x1, x0, 4
    addi x3, x1, 8
>>>>>>> afca87bfcbc203ffe79bc44ea0881a25e6bbe9dc

    slti x0, x0, -256 # this is the magic instruction to end the simulation
    nop               # preventing fetching illegal instructions
    nop
    nop
    nop
    nop
