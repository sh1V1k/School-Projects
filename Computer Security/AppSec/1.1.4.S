.section .data
    str: .string "coolfood"

.section .text
.global your_asm_fn

your_asm_fn:

push    %ebp
mov     %esp,%ebp

# Your code here
#get pointer to int value and store in eax
push %eax
push $0xc105ed
mov %esp, %eax

push $str #string which is passed as a pointer to char array
push %eax #int pointer
push $0xacce55ed #int
call practice

leave
ret
