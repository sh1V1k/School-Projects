.section .data
    shell: .string "/bin/sh"



.global _main
.section .text

_main:

push    %ebp
mov     %esp,%ebp

# Your code here
xorl %edx, %edx # null value that we will use later
mov $11, %eax #sys call number
mov $shell, %ebx #file path
push %edx
push $shell
xorl %ecx, %ecx #args
 # don't have to put null in edx since it already contains it
int $0x80

leave
ret
