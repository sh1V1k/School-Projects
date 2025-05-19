#pragma once
#ifndef SCHEDULE_H
#define SCHEDULE_H

#include "i8259.h"
#include "types.h"
#include "paging.h"
#include "syscall.h"
#include "schedule_asm.h"


typedef struct terminal_struct{ //file descriptors
    char terminal_buffer[BUFFER_SIZE-1];
    int cursor_x;
    int cursor_y;
    int cur_idx;
    volatile int read_flag;
    uint32_t old_mapping;
    uint32_t ebp;
    uint32_t esp;
}terminal_t;

terminal_t terminals[3];


void schedule();
void switch_terminal(uint8_t terminal_num);
void terminal_init(uint8_t terminal_num);
int get_active();
int get_process();



#endif /* SYSCALL_H */
