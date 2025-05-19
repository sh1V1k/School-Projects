#include "syscall.h"
#include "lib.h"
#include "file_system.h"
#include "paging.h"
#include "paging_asm.h"
#include "syscall_asm.h"
#include "schedule.h"

#define BIT_MASK 0xFFFFE000 // used to isolate esp
// WEEK 1


static int pcb_bitmap[MAX_TERMINALS][MAX_PCB_AMOUNT] = {{0}};
static int base_shell = -1;
static int old_pcb_amount = 0;

// volatile pcb_t* cur_pcb;

/* Execute a valid given command*/
/* execute(const uint8_t* command)
 *
 * Used to call and execute user programs. Will be used in kernel.c to execute the shell
 * to begin our OS in user-space rather than kernel space. It is further used will we are
 * in the shell to execute other user programs like cat, ls, etc.
 *
 * Inputs: command is a space-separated sequence of words. The first word is the file name of the
 *         program to be executed, and the rest of the command—stripped of leading spaces—should be provided to the new
 *         program on request via the getargs system call
 *
 * Outputs: -1 if command cannot be executed(if program does not exist or filename specified is not an executable)
 *          256 if the program dies by an exception, or a value from range 0 - 255 if program executes a halt system call
 * Side Effects: Will switch between user space and kernel space depending on the command
 */
int32_t execute(const uint8_t *command)
{
    cli();
    // Parse arguments (cat frame1.txt)
    uint32_t i;
    uint32_t len_first_arg = 0; // number of bytes we should copy
    uint32_t start_second_arg = 0;
    uint32_t len_second_arg = 0;
    int8_t cmd[FILE_NAME_LEN] = {'\0'}; // 9 = longest executable name
    uint8_t file_data[40];
    int8_t user_arg[USER_ARG_LEN] = {'\0'}; // buffer to store user inputs
    uint32_t size = strlen((const int8_t *)command);
    dentry_t temp_dentry; // used to read the data of the executable file
    inode_t *cur_inode;
    uint32_t eip = 0;
    uint32_t ebp = 0;
    uint32_t esp = 0;
    uint32_t physical_address;
    pcb_t *process_block;

    for (i = 0; i < size; i++)
    {
        if (command[i] == ' ' || command[i] == '\0')
        {
            break;
        }
        len_first_arg++;
    }

    if(len_first_arg > 9){ return -1; } // 9 = longest executable name

    for (i = len_first_arg + 1; i < size; i++)
    {
        if (command[i] == ' ' || command[i] == '\0')
        {
            break;
        }
        len_second_arg++;
    }

    start_second_arg = (uint32_t)((int8_t *)command + len_first_arg + 1);
    // for(i = 0; i < USER_ARG_LEN; i++){
    //     user_arg[i] = '\0';
    // }
    (void)strncpy((int8_t *)cmd, (int8_t *)command, len_first_arg);
    (void)strncpy((int8_t *)user_arg, (int8_t *)start_second_arg, len_second_arg + 1);

    // Executable check (cat is an executable)
    if (read_dentry_by_name((const uint8_t *)cmd, &temp_dentry) != 0)
    {
        return -1;
    }
    cur_inode = (inode_t *)((uint32_t)first_inode + (FILE_SEGMENT)*temp_dentry.inode_num);
    if (read_data(temp_dentry.inode_num, 0, file_data, 40) != 40)
    {
        
        return -1;
    }
    if (file_data[0] != ELF_MAGIC_0 || file_data[1] != ELF_MAGIC_1 || file_data[2] != ELF_MAGIC_2 || file_data[3] != ELF_MAGIC_3)
    {
        return -1;
    }
    // 24-27 stores EIP in file data
    eip |= file_data[27] << 24; // Little endian-stored
    eip |= file_data[26] << 16;
    eip |= file_data[25] << 8;
    eip |= file_data[24];

    // Check if we reached the maximum supported PCB’s (Max programs).
    if (pcb_amount >= MAX_PCB_AMOUNT)
    {
        return -1;
    }
    if(base_shell == -1){
        physical_address = (MB_HEX << 3) + pcb_amount * (MB_HEX << 2); // MAGIC
    if(pcb_amount < 3){ pcb_bitmap[get_process()][pcb_amount] = 1; }
    else{ pcb_bitmap[get_active()][pcb_amount] = 1; }
    pcb_amount++;

    // Set up program paging (4MB) // how do we actually load the file into this 4mb page?
    page_dir_4mb_t process_page;
    process_page.avl = 0;                                     // extra bits ?
    process_page.ps = 1;                                      // page size - 1: 4mb, 0: 4kb
    process_page.a = 0;                                       // accessed - 1: Yes, 0: No
    process_page.pcd = 0;                                     // page cached
    process_page.pwt = 0;                                     // page write-though
    process_page.us = 1;                                      // user/supervisor mode - permission mode
    process_page.rw = 1;                                      // read/write
    process_page.p = 1;                                       // Present
    process_page.pat = 0;                                     // Page Attribute table
    process_page.d = 0;                                       // dirty
    process_page.g = 1;                                       // global
    process_page.base_addr = physical_address >> 22;          // kernel address right shifted by 12
    page_dir[_128MB_ >> 22] = (uint32_t)(process_page.entry); // MAGIC - 32

    // flush cache
    asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
    ");
    // User-level Program Loader (load from FS to program page)

    (void)read_data((uint32_t)temp_dentry.inode_num, (uint32_t)0, (uint8_t *)PROCESS_VIRTUAL_PAGE, (uint32_t)cur_inode->data_len);

    // Starts at 8MB - 8KB * pcb number
    process_block = (pcb_t *)((MB_HEX << 3) - (2 * FILE_SEGMENT * pcb_amount));

    // Create Process Control Block (PCB)
    // page faulting here for second process;
    process_block->pid = pcb_amount - 1;
    process_block->open_file_idx = 2;

    // process_block->open_files[0].file_operation = stdin;
    process_block->open_files[0].file_operation = stdin;
    process_block->open_files[0].f_flags = 1;
    process_block->open_files[0].f_inode = -1; // null/should not be used
    process_block->open_files[0].f_pos = -1;

    // process_block->open_files[1].file_operation = stdout;
    process_block->open_files[1].file_operation = stdout;
    process_block->open_files[1].f_flags = 1;
    process_block->open_files[1].f_inode = -1; // null/should not be used
    process_block->open_files[1].f_pos = -1;

    // (void)strncpy((int8_t *)process_block->user_args, user_arg, strlen(user_arg)); // user inputs
    (void)strncpy((int8_t *)process_block->user_args, user_arg, USER_ARG_LEN);
    //// edit ESP0 in TSS which should contain process’s kernel-mode stack, maybe need to modify ss0 to kernel stack(4MB)
    tss.ss0 = KERNEL_DS;
    tss.esp0 = (MB_HEX << 3) - (2 * FILE_SEGMENT * (pcb_amount - 1)) - 4;
    process_block->ss0 = tss.ss0;
    process_block->esp0 = tss.esp0;

    asm("\t movl %%ebp, %0" : "=r"(ebp));
    process_block->ebp = ebp;
    asm("\t movl %%esp, %0" : "=r"(esp));
    process_block->esp = esp;
    process_block->eip = eip;

    // Context Switch
    // Create its own context switch stack
    // IRET (and then you will start running user’s code)
    sti();
    context_switch(USER_DS, 0x8400000 - 4, USER_CS, eip);
    return 0;
    } else{
        old_pcb_amount = pcb_amount;
        pcb_amount = base_shell;
    physical_address = (MB_HEX << 3) + pcb_amount * (MB_HEX << 2); // MAGIC
    if(pcb_amount < 3){ pcb_bitmap[get_process()][pcb_amount] = 1; }
    else{ pcb_bitmap[get_active()][pcb_amount] = 1; }
    pcb_amount++;

    // Set up program paging (4MB) // how do we actually load the file into this 4mb page?
    page_dir_4mb_t process_page;
    process_page.avl = 0;                                     // extra bits ?
    process_page.ps = 1;                                      // page size - 1: 4mb, 0: 4kb
    process_page.a = 0;                                       // accessed - 1: Yes, 0: No
    process_page.pcd = 0;                                     // page cached
    process_page.pwt = 0;                                     // page write-though
    process_page.us = 1;                                      // user/supervisor mode - permission mode
    process_page.rw = 1;                                      // read/write
    process_page.p = 1;                                       // Present
    process_page.pat = 0;                                     // Page Attribute table
    process_page.d = 0;                                       // dirty
    process_page.g = 1;                                       // global
    process_page.base_addr = physical_address >> 22;          // kernel address right shifted by 12
    page_dir[_128MB_ >> 22] = (uint32_t)(process_page.entry); // MAGIC - 32

    // flush cache
    asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
    ");
    // User-level Program Loader (load from FS to program page)

    (void)read_data((uint32_t)temp_dentry.inode_num, (uint32_t)0, (uint8_t *)PROCESS_VIRTUAL_PAGE, (uint32_t)cur_inode->data_len);

    // Starts at 8MB - 8KB * pcb number
    process_block = (pcb_t *)((MB_HEX << 3) - (2 * FILE_SEGMENT * pcb_amount));

    // Create Process Control Block (PCB)
    // page faulting here for second process;
    process_block->pid = pcb_amount - 1;
    process_block->open_file_idx = 2;

    // process_block->open_files[0].file_operation = stdin;
    process_block->open_files[0].file_operation = stdin;
    process_block->open_files[0].f_flags = 1;
    process_block->open_files[0].f_inode = -1; // null/should not be used
    process_block->open_files[0].f_pos = -1;

    // process_block->open_files[1].file_operation = stdout;
    process_block->open_files[1].file_operation = stdout;
    process_block->open_files[1].f_flags = 1;
    process_block->open_files[1].f_inode = -1; // null/should not be used
    process_block->open_files[1].f_pos = -1;

    // (void)strncpy((int8_t *)process_block->user_args, user_arg, strlen(user_arg)); // user inputs
    (void)strncpy((int8_t *)process_block->user_args, user_arg, USER_ARG_LEN);
    //// edit ESP0 in TSS which should contain process’s kernel-mode stack, maybe need to modify ss0 to kernel stack(4MB)
    tss.ss0 = KERNEL_DS;
    tss.esp0 = (MB_HEX << 3) - (2 * FILE_SEGMENT * (pcb_amount - 1)) - 4;
    process_block->ss0 = tss.ss0;
    process_block->esp0 = tss.esp0;

    asm("\t movl %%ebp, %0" : "=r"(ebp));
    process_block->ebp = ebp;
    asm("\t movl %%esp, %0" : "=r"(esp));
    process_block->esp = esp;
    process_block->eip = eip;

    // Context Switch
    // Create its own context switch stack
    // IRET (and then you will start running user’s code)
    pcb_amount = old_pcb_amount;
    base_shell = -1;
    sti();
    context_switch(USER_DS, 0x8400000 - 4, USER_CS, eip);
    }
    return -1;
}

/* Used to terminate a process*/
/* halt(uint8_t status)
 *
 * The system call handler itself is responsible for expanding the 8-bit argument from BL into the 32-bit return value to the parent program’s
 * execute system call. Be careful not to return all 32 bits from EBX
 *
 * Inputs: status
 *
 * Outputs: SHOULD NEVER RETURN TO THE CALLER
 * Side Effects: NONE/IDK
 */
int32_t halt(uint8_t status)
{
    cli();
    int i;
    //uint32_t ebp = 0;
    pcb_t *cur_pcb = get_pcb();
    pcb_t *parent_pcb = get_parent(cur_pcb->pid);

    if (cur_pcb->pid > 2 /* && cur_pcb.name != "shell"*/) //? if not shell - might need changing
    {
        // todo - set cur_page to parent_pcb context
        // todo - context switch to parent_pcb

        /*//? reset paging */ // TODO:
        uint32_t physical_address = (MB_HEX << 3) + parent_pcb->pid * (MB_HEX << 2);
        page_dir_4mb_t process_page;
        process_page.avl = 0;                                     // extra bits ?
        process_page.ps = 1;                                      // page size - 1: 4mb, 0: 4kb
        process_page.a = 0;                                       // accessed - 1: Yes, 0: No
        process_page.pcd = 0;                                     // page cached
        process_page.pwt = 0;                                     // page write-though
        process_page.us = 1;                                      // user/supervisor mode - permission mode
        process_page.rw = 1;                                      // read/write
        process_page.p = 1;                                       // Present
        process_page.pat = 0;                                     // Page Attribute table
        process_page.d = 0;                                       // dirty
        process_page.g = 1;                                       // global
        process_page.base_addr = physical_address >> 22;          // kernel address right shifted by 12
        page_dir[_128MB_ >> 22] = (uint32_t)(process_page.entry); // MAGIC - 32

        // flush the cache
        asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
        ");

        /*close all fd*/
        for (i = 0; i < MAX_OPEN_FILES; i++)
        {
            (void)close(i);
        }

        tss.ss0 = KERNEL_DS;
        tss.esp0 = parent_pcb->esp0; // page fault baybeeeee

        // in_use_pcb[cur_pcb->pid] = 0; // set pid as available //!might want to establish check in execute for mp5
        pcb_amount--;                 // temporary pcb tracking to match current solution
        pcb_bitmap[get_process()][cur_pcb->pid] = 0;

    }
    else
    {
        // in_use_pcb[cur_pcb->pid] = 0;
        // asm("\t movl %%ebp, %0" : "=r"(ebp));
        base_shell = cur_pcb->pid;
        (void)execute((const uint8_t*)"shell");
        // pcb_amount = 0;
        sti();
        
    }
    sti();
    return_from_halt(cur_pcb->ebp, status);

    return -1;
}
/* read(uint8_t status)
 *
 * Syscall for read
 *
 * Inputs: int32_t fd, void *buf, int32_t nbytes
 *
 * Outputs: number of bytes
 * Side Effects: call a read function
 */
int32_t read(int32_t fd, void *buf, int32_t nbytes)
{
    /*
     *  Call correct read function
        How to determine correct read?
    */
    int32_t temp_ret = 0;
    pcb_t *cur_pcb = get_pcb();
    if (fd < 0 || fd > MAX_OPEN_FILES - 1)
    {
        return -1;
    }
    if (fd == 1)
    {
        return -1;
    }
    if (cur_pcb->open_files[fd].f_flags != 0)
    {
        if ((temp_ret = cur_pcb->open_files[fd].file_operation.read(fd, buf, nbytes)) == 0)
        {
            return 0;
        }
        else {
            return temp_ret;
        }
    }
    else
    {
        return -1;
    }
    //return nbytes;
}

/* write(int32_t fd, const void *buf, int32_t nbytes)
 *
 * Syscall for write
 *
 * Inputs: int32_t fd, const void *buf, int32_t nbytes
 *
 * Outputs: number of bytes
 * Side Effects: calls a write function
 */
int32_t write(int32_t fd, const void *buf, int32_t nbytes)
{
    // Should be same as read accept calling write function instead
    // puts((char*)buf);
    pcb_t *cur_pcb = get_pcb();
    if (fd < 0 || fd > MAX_OPEN_FILES - 1)
    {
        return -1;
    }
    if (fd == 0)
    {
        return -1;
    }
    if (cur_pcb->open_files[fd].f_flags != 0)
    {
        cur_pcb->open_files[fd].file_operation.write(fd, buf, nbytes);
    }
    else
    {
        return -1;
    }

    return nbytes;
}
/* open(const uint8_t *filename)
 *
 * Syscall for open
 *
 * Inputs: int32_t fd, const void *buf, int32_t nbytes
 *
 * Outputs: number of bytes
 * Side Effects: calls a open function
 */
int32_t open(const uint8_t *filename)
{

    dentry_t temp;
    pcb_t *cur_pcb = get_pcb();
    if (read_dentry_by_name(filename, &temp) != -1)
    {
        if (cur_pcb->open_file_idx < 2 || cur_pcb->open_file_idx > MAX_OPEN_FILES - 1)
        {
            return -1;
        }
        if (temp.filetype == 0) //rtc 
        {
            cur_pcb->open_files[cur_pcb->open_file_idx].f_inode = 0;
            cur_pcb->open_files[cur_pcb->open_file_idx].file_operation = rtc;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_flags = 1;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_pos = 0;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_filetype = 0;
        }
         else if (temp.filetype == 1) // dir
        {
            cur_pcb->open_files[cur_pcb->open_file_idx].f_inode = 0;
            cur_pcb->open_files[cur_pcb->open_file_idx].file_operation = dir;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_flags = 1;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_pos = 0;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_filetype = 1;
        }

        else if (temp.filetype == 2) // file
        {
            cur_pcb->open_files[cur_pcb->open_file_idx].f_inode = temp.inode_num;
            cur_pcb->open_files[cur_pcb->open_file_idx].file_operation = file;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_flags = 1;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_pos = 0;
            cur_pcb->open_files[cur_pcb->open_file_idx].f_filetype = 2;
        }
        else { // other
            cur_pcb->open_files[cur_pcb->open_file_idx].f_flags = 1;
        }
        // Set open_file_idx
        while (cur_pcb->open_file_idx != 8 && cur_pcb->open_files[cur_pcb->open_file_idx].f_flags == 1)
        {
            cur_pcb->open_file_idx++;
        }
    }
    else
    {
        return -1;
    }
    return cur_pcb->open_file_idx - 1;
}
/* close(int32_t fd))
 *
 * Syscall for close
 *
 * Inputs: int32_t fd
 *
 * Outputs: number of bytes
 * Side Effects: calls a open function
 */
int32_t close(int32_t fd)
{
    pcb_t *cur_pcb = get_pcb();
    int i;
    if (fd < 2 || fd > MAX_FILE_OPEN - 1)
    {
        return -1;
    }
    if (cur_pcb->open_files[fd].f_flags == 0)
    {
        return -1;
    } // does nothing the file is not in use
    else
    {                                        // file in use and we want to close it
        cur_pcb->open_files[fd].f_flags = 0; // set to not in use and update idx
        for (i = 0; i < MAX_OPEN_FILES; i++)
        {
            if (cur_pcb->open_files[i].f_flags == 0)
            {
                cur_pcb->open_file_idx = i;
                break;
            }
        }
    }
    return 0;
}

// IGNORE: THIS IS FOR CHECKPOINT 4
//
//

/* getargs(uint8_t *buf, int32_t nbytes)
 *
 * Gets arguments
 *
 * Inputs: int32_t fd
 *
 * Outputs: 0
 * Side Effects: does fun stuff
 */
int32_t getargs(uint8_t *buf, int32_t nbytes)
{
    pcb_t *cur_pcb = get_pcb();

    uint32_t arg_len = strlen((const int8_t *)cur_pcb->user_args) + 1; // plus one to account for the null character at the end
    
    // int i;
    // for(i = 0; i < USER_ARG_LEN; i++){
    //     ((char*)buf)[i] = '\0';
    // }
    if ((arg_len == 0) || (arg_len > FILE_NAME_LEN + 1))
    {
        return -1;
    }
    else
    {
        (void)strncpy((int8_t *)buf, (int8_t *)cur_pcb->user_args, (uint32_t)nbytes);
    }

    return 0;
}

/* vidmap(uint8_t** screen_start)
 *
 * maps the video memory to a page in virtual memory
 *
 * Inputs: uint8_t** screen_start - video memory
 *
 * Outputs: 0
 * Side Effects: creates a new page to be virtual video memory
 */

int32_t vidmap(uint8_t **screen_start)
{
    // printf("in vidmap");
    cli();
    if (screen_start == NULL || (uint32_t)(screen_start) < _128MB_ || (uint32_t)(screen_start) >= _128MB_ + (MB_HEX * 4))
    {
        return -1;
    }
    

    page_dir_4kb_t vidmem;
    vidmem.p = 1;
    vidmem.rw = 1;
    vidmem.us = 1;
    vidmem.pwt = 0;
    vidmem.pcd = 0;
    vidmem.a = 0;
    vidmem.reserved = 0;
    vidmem.ps = 0;
    vidmem.g = 0;
    vidmem.avl = 0;
    vidmem.offset_32_12 = (uint32_t)(vidmap_page_tab) >> 12;
    page_dir[((_128MB_ + (MB_HEX * 4)) >> 22) & 0x3ff] = (uint32_t)(vidmem.entry); 
    // use upper 10 bits to index into page directory (index is 33, one after first process which is 32)

    page_tab_t vidmem_tab_entry;
    vidmem_tab_entry.p = 1;
    vidmem_tab_entry.rw = 1;
    vidmem_tab_entry.us = 1;
    vidmem_tab_entry.pwt = 0;
    vidmem_tab_entry.pcd = 0;
    vidmem_tab_entry.a = 0;
    vidmem_tab_entry.d = 0;
    vidmem_tab_entry.pat = 0;
    vidmem_tab_entry.g = 0;
    vidmem_tab_entry.avl = 0;
    vidmem_tab_entry.offset_32_12 = 0xb8;                                                           // 0xb8 = 0xb8000 >> 12 which is user vid mem
    vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = (uint32_t)(vidmem_tab_entry.entry); // use middle 10 bits to address into page table (index is 0)
    // first_page_tab[0xb8] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + get_active()))>>12];

    // flush the cache
    asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
        ");

    // 1000 0100 0000 0000 0000 0000 0000 0000
    // uint32_t temp = _128MB_ + (MB_HEX * 4);
    *screen_start = (uint8_t *)(_128MB_ + (MB_HEX * 4));
    sti();
    return 0;
}
int32_t set_handler(int32_t signum, void *handler_address)
{
    printf("in set_handler");

    return 0;
}
int32_t sigreturn(void)
{
    printf("in sigreturn");
    return 0;
}
/* get_pcb_inline(void)
 *
 * gets pcb
 *
 * Inputs: 
 *
 * Outputs: 0
 * Side Effects: none
 */
pcb_t *get_pcb_inline()
{

    pcb_t *pcb_ptr;

    asm volatile("                 \n\
            andl  %%esp,%%ebx       \n\
            movl   %%ebx, %%eax     \n\
            "
                 : "=a"(pcb_ptr)
                 : "b"(BIT_MASK));

    return pcb_ptr;
}
/* get_pcb()
 *
 * gets current pcb
 *
 * Inputs: 
 *
 * Outputs: PCB
 * Side Effects: none
 */
pcb_t *get_pcb()
{  
    int i;
    for(i = MAX_PCB_AMOUNT-1; i >= 0; i--){ if( (pcb_bitmap[get_process()][i]) == 1){ break; } }
    return (pcb_t *)((MB_HEX << 3) - (2 * FILE_SEGMENT * (i+1)));
}
/* get_pcb_active()
 *
 * gets current active pcb
 *
 * Inputs: None
 *
 * Outputs: PCB
 * Side Effects: none
 */
pcb_t* get_pcb_active(){
    int i;
    for(i = MAX_PCB_AMOUNT-1; i >= 0; i--){ if(pcb_bitmap[get_active()][i] == 1){ break; } }
    return (pcb_t *)((MB_HEX << 3) - (2 * FILE_SEGMENT * (i+1)));
}

/* get_parent(uint32_t pid)
 *
 * gets parent pcb of current pid
 *
 * Inputs: pid we are searching for the parent of
 *
 * Outputs: PCB
 * Side Effects: none
 */
pcb_t* get_parent(uint32_t pid){
    int i;
    for(i = pid-1; i >= 0; i--){ if(pcb_bitmap[get_process()][i] == 1){ break; } }
    return (pcb_t *)((MB_HEX << 3) - (2 * FILE_SEGMENT * (i+1)));
}

/* terminal_read_null(int32_t fd, void *buffer, int32_t number_bytes)
 *
 * Returns error (-1)
 */
int32_t terminal_read_null(int32_t fd, void *buffer, int32_t number_bytes)
{
    return -1;
}

/* terminal_write_null(int32_t fd, const void *buf, int32_t number_bytes)
 *
 * Returns error (-1)
 */
int32_t terminal_write_null(int32_t fd, const void *buf, int32_t number_bytes)
{
    return -1;
}
