#pragma once
#ifndef SYSCALL_H
#define SYSCALL_H

#include "types.h"
#include "idt.h"
#include "terminal.h"
#include "x86_desc.h"
#include "rtc.h"
#include "file_system.h" 
#include "syscall_asm.h"
//all valid executable files: cat grep hello ls pingpong counter shell sigtest testprint syserr

#define ELF_MAGIC_0  0x7f
#define ELF_MAGIC_1  0x45
#define ELF_MAGIC_2  0x4c
#define ELF_MAGIC_3  0x46
#define MAX_TERMINALS 3
#define MAX_PCB_AMOUNT 6
#define MAX_OPEN_FILES 8
#define MB_HEX 0x100000
#define PROCESS_VIRTUAL_PAGE 0x08048000
#define _128MB_ 0x8000000
#define USER_ARG_LEN 128


typedef struct file_operations{ //jump table for function pointers
  int32_t (*open)(const uint8_t *fname);
  int32_t (*close)(int32_t fd); 
  int32_t (*read)(int32_t fd, void *buf, int32_t nbytes);
  int32_t (*write)(int32_t fd, const void* buf, int32_t nbytes);
}fo_t;


typedef struct file_struct{ //file descriptors
  fo_t file_operation; //associated jump tables
  int32_t f_inode; //inode num associated with that file
  int32_t f_pos; //current position in the file, updated by file read
  int32_t f_flags; //1 for in use 0 for not in use
  int32_t f_filetype; // some repeated info, stores name and file_type, inode_num duplicate
}fds_t;

typedef struct process_block{
    uint32_t pid;
    fds_t open_files[MAX_OPEN_FILES]; 
    int32_t open_file_idx;
    uint32_t ebp;
    uint32_t esp;
    uint32_t eip;
    int32_t ss0;
    int32_t esp0;
    uint8_t user_args[USER_ARG_LEN];
}pcb_t;

fo_t stdin;
fo_t stdout;
fo_t file;
fo_t dir;
fo_t rtc;
volatile uint32_t pcb_amount;

uint32_t in_use_pcb[MAX_PCB_AMOUNT];

extern int32_t halt(uint8_t status);
extern int32_t execute(const uint8_t* command);
extern int32_t read(int32_t fd, void* buf, int32_t nbytes);
extern int32_t write(int32_t fd, const void* buf, int32_t nbytes);
extern int32_t open(const uint8_t* filename);
extern int32_t close(int32_t fd);
extern int32_t getargs(uint8_t* buf, int32_t nbytes);
extern int32_t vidmap(uint8_t** screen_start);
extern int32_t set_handler(int32_t signum, void* handler_address);
extern int32_t sigreturn(void);
extern pcb_t* get_pcb();
extern pcb_t* get_pcb_active();
extern pcb_t* get_parent();
extern pcb_t* get_pcb_inline();
extern void syscall();
extern int32_t terminal_read_null(int32_t fd, void* buffer, int32_t number_bytes);
extern int32_t terminal_write_null(int32_t fd, const void* buf, int32_t number_bytes);

#endif /* SYSCALL_H */

