.section .text
.globl _start
_start:

    auipc   x1 , 0
    lui     x2 , 0xAA55A
    addi    x3 , x1, 1
    add     x4 , x1, x2
    lw      x5, 4(x1)
    sw      x2, 0(x1)
    lw      x6, some_data_1
    # la      x6, some_data_1
    # lw      x6, 0(x6)
    sw      x6, some_data_2, x7
    bne     x1, x2, end
    nop

end:

    slti x0, x0, -256

.section .data
some_data_1:
    .word   0xAA552333
some_data_2:
    .half   0x0000
    .byte   0x00
    .byte   0x00
