.global your_asm_fn
.section .text

your_asm_fn:

push    %ebp
mov     %esp,%ebp

# Your code here
push $0xc001f00d
# esp will contain the address(pointer to our value which we can put in eax)
mov %esp, %eax
push %eax
push $0xc105ed
push $0xacce55ed
call practice

leave
ret
