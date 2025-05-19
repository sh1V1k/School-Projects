.global your_asm_fn
.section .text

your_asm_fn:

push    %ebp
mov     %esp,%ebp

# Your code here
# x86 pushes arguements from right to left
push $0xc001f00d
push $0xc105ed
push $0xacce55ed
# call push eip which is the return address
call practice


leave
ret
