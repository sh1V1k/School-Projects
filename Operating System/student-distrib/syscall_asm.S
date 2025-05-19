#define ASM 1

#define CONTEXT_SWITCH(ss, esp, cs, eip) 
.globl context_switch
context_switch:
    movl %esp, %ecx
    
    movw 4(%ecx), %ax
    movw %ax, %ds
    pushl 4(%ecx)                                              
    pushl 8(%ecx)                                               
    pushfl
    popl %edx
    orl $0x0200, %edx
    pushl %edx
    pushl 12(%ecx)
    pushl 16(%ecx)
    
    iret



#define RETURN_FROM_HALT(ebp,status)
.globl return_from_halt 
return_from_halt:
    movl    8(%esp),%eax   #3rd arg (status)
    movl    4(%esp),%ebp   #2nd arg (esp)
    movl    %ebp, %esp   

    popl %ebp
    ret


.align 4
sys_jmp_table: // jump table of syscall handlers
    .long 0 // syscall handlers are 1-indexed
    .long halt
    .long execute
    .long read
    .long write
    .long open
    .long close
    .long getargs
    .long vidmap
    .long set_handler
    .long sigreturn

.align 4
.globl syscall
syscall:
    pushl %edi // eax is edited so manually save all registers
    pushl %esi
    pushl %ebx
    pushl %ebp
    pushl %ecx
    pushl %edx
    pushl %esp
    pushfl

    // cmpl $0, %eax // bounds check on syscall number (syscalls are 1-10 inclusive)
    // jle syscall_error
    cmpl $0, %eax
    je syscall_error
    cmpl $10, %eax
    ja syscall_error

    pushl %edx // push arguments for syscalls
    pushl %ecx
    pushl %ebx

    call *sys_jmp_table(, %eax, 4)
    addl $12, %esp // pop arguments off stack
    jmp syscall_done

    syscall_error:
        movl $-1, %eax
    syscall_done:
        popfl
        popl %esp
        popl %edx
        popl %ecx
        popl %ebp
        popl %ebx
        popl %esi
        popl %edi
        iret

