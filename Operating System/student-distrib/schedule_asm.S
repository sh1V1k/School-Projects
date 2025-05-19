#define ASM 1

#define RETURN_SCHEDULE(ebp,esp)
.globl return_schedule 
return_schedule:
    movl    4(%esp),%ebp   #1st arg (ebp)
    movl    8(%esp),%esp   #2nd arg (esp)    

    leave
    ret

