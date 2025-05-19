#pragma once
#include "x86_desc.h"
#include "handlers.h"
#include "lib.h"



#define SYSCALL_VEC 0x80 //maybe be x80 or 128
#define KEYBOARD_VEC 0x21 
#define RTC_VEC 0x28
#define PIT_VEC 0x20

//struct for registers
struct x86_regs {
    uint32_t eflags;
    uint32_t eip;
    uint32_t edi;
    uint32_t esi;
    uint32_t ebp;
    uint32_t esp;
    uint32_t ebx;
    uint32_t edx;
    uint32_t ecx;
    uint32_t eax;
 } __attribute__ (( packed ));

void idt_init();
void handler(uint32_t id, uint32_t fl, struct x86_regs regs, uint32_t err);

