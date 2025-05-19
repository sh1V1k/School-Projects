#define ASM 1



cr0_mask:
    .long 0x80000001
four_mb_cr4_mask:
    .long 0x00000010
#define ENABLE_PAGING(directory) 
.globl enable_paging
enable_paging:
    pushl %ebp /* from enabling paging section in "Setting Up Paging" osdev*/
    movl %esp, %ebp
    movl 8(%ebp), %eax
    movl %eax, %cr3
    
    movl %cr4, %eax
    orl four_mb_cr4_mask, %eax /*enables 4mb paging*/
    movl %eax, %cr4

    movl %cr0, %eax
    orl cr0_mask, %eax /*sets paging bit and protection bit */
    movl %eax, %cr0

    movl %cr3, %eax
    movl %eax, %cr3 /*flush tlb*/

    leave
    ret

