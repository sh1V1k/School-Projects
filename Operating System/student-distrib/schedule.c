#include "schedule.h"
#include "syscall.h"
#include "paging.h"

// GLOBAL VARIABLES
int active_terminal = 0; // the terminal being shown to the user
int terminal1 = 0;
int terminal2 = 0;
int terminal3 = 0;
int process_terminal = 0; // the terminal being processed by schedule



/* Schedule */
/* schedule()
 *
 * Sets up scheduling for MAX_TERMINALS
 *
 * Inputs: None
 * Outputs: None
 * Side Effects: Terminal processes run in round-robin schedule
 */
void schedule()
{
    send_eoi(0);
    if (pcb_amount != 0)
    {
        // round robin style
        uint32_t ebp;
        uint32_t esp;
        asm("\t movl %%ebp, %0" : "=r"(ebp));
        terminals[process_terminal].ebp = ebp;
        asm("\t movl %%esp, %0" : "=r"(esp));
        terminals[process_terminal].esp = esp;

        // terminals[process_terminal].cursor_x = getScreenX();
        // terminals[process_terminal].cursor_y = getScreenY();

        process_terminal = (process_terminal + 1) % MAX_TERMINALS;

        page_tab_t temp;
        page_tab_t temp2;
        if (get_process() == get_active())
        {
            temp.entry = first_page_tab[0xb8];
            temp.offset_32_12 = 0xb8;
            first_page_tab[0xb8] = (uint32_t)temp.entry;
            // first_page_tab[0xb8] = first_page_tab[0xbc];
            temp2.entry = vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff];
            temp2.offset_32_12 = 0xb8;
            vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = (uint32_t)temp2.entry;
            // vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = first_page_tab[0xbc];
        }
        else
        {
            temp.entry = first_page_tab[0xb8];
            temp.offset_32_12 = (VIDEO_MEM + (0x1000) * (1 + get_process())) >> 12;
            first_page_tab[0xb8] = (uint32_t)temp.entry;
            // first_page_tab[0xb8] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + get_process()))>>12];
            // vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + process_terminal))>>12];
            temp2.entry = vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff];
            temp2.offset_32_12 = (VIDEO_MEM + (0x1000) * (1 + get_process())) >> 12;
            vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = (uint32_t)temp2.entry;
            // vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + get_process()))>>12];
        }

        asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
        ");

        // initiliaze base terminals
        if (!terminal2)
        {
            terminal2++;
            setScreenX(terminals[process_terminal].cursor_x);
            setScreenY(terminals[process_terminal].cursor_y);
            (void)execute((const uint8_t *)"shell");
        }
        if (!terminal3)
        {
            terminal3++;
            setScreenX(terminals[process_terminal].cursor_x);
            setScreenY(terminals[process_terminal].cursor_y);
            (void)execute((const uint8_t *)"shell");
        }

        pcb_t *switch_process = get_pcb();

        uint32_t physical_address = (MB_HEX << 3) + switch_process->pid * (MB_HEX << 2);
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

        tss.ss0 = KERNEL_DS;
        tss.esp0 = switch_process->esp0;

        setScreenX(terminals[process_terminal].cursor_x);
        setScreenY(terminals[process_terminal].cursor_y);

        return_schedule(terminals[process_terminal].ebp, terminals[process_terminal].esp);
    }
}

/* Switch Terminal */
/* switch_terminal(uint8_t terminal_num)
 *
 * Switches to terminal_num terminal
 *
 * Inputs: terminal_num: terminal to switch to
 * Outputs: None
 * Side Effects: Switches current terminal
 */
void switch_terminal(uint8_t terminal_num)
{
    cli();
    int old_active = active_terminal;
    active_terminal = terminal_num;
    page_tab_t temp;
    page_tab_t temp2;
    if (get_process() == active_terminal)
    {
        temp.entry = first_page_tab[0xb8];
        temp.offset_32_12 = 0xb8;
        first_page_tab[0xb8] = (uint32_t)temp.entry;
        temp2.entry = vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff];
        temp2.offset_32_12 = 0xb8;
        vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = (uint32_t)temp2.entry;
    }
    else
    {
        temp.entry = first_page_tab[0xb8];
        temp.offset_32_12 = (VIDEO_MEM + (0x1000) * (1 + get_process())) >> 12;
        first_page_tab[0xb8] = (uint32_t)temp.entry;
        // first_page_tab[0xb8] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + get_process()))>>12];
        // vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + process_terminal))>>12];
        temp2.entry = vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff];
        temp2.offset_32_12 = (VIDEO_MEM + (0x1000) * (1 + get_process())) >> 12;
        vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = (uint32_t)temp2.entry;
        // vidmap_page_tab[((_128MB_ + (MB_HEX * 4)) >> 12) & 0x3ff] = first_page_tab[(VIDEO_MEM + (0x1000)*(1 + get_process()))>>12];
    }
    // first_page_tab[0xb8] = old_vid_mapping;

    asm volatile("      \n\
        movl %cr3, %eax \n\
        movl %eax, %cr3 \n\
        ");

    memcpy(terminals[old_active].terminal_buffer, keyboard_buf, BUFFER_SIZE - 1);

    // save video memory to video page for associated terminal
    memcpy((void*)(VIDEO_MEM + (0x1000) * (1 + old_active)), (void*)0xBC000, FILE_SEGMENT);

    // restore video memory of switch to terminal
    memcpy((void*)0xBC000, (void*)(VIDEO_MEM + (0x1000) * (1 + active_terminal)), FILE_SEGMENT);

    setScreenX(terminals[active_terminal].cursor_x);
    setScreenY(terminals[active_terminal].cursor_y);
    update_cursor();
    memcpy(keyboard_buf, terminals[active_terminal].terminal_buffer, BUFFER_SIZE - 1);

    /*// first_page_tab[(VIDEO_MEM + (0x1000)*(1 + old_active))>>12] = current_mappings;
    // asm volatile("      \n\
    //     movl %cr3, %eax \n\
    //     movl %eax, %cr3 \n\
    //     ");
    */
    sti();
}

/* Get Active */
/* get_active()
 *
 * Returns active terminal
 *
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 */
int get_active()
{
    return active_terminal;
}

/* Get Process */
/* get_process()
 *
 * Returns process terminal
 *
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 */
int get_process()
{
    return process_terminal;
}
