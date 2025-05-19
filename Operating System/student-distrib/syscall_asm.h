#pragma once
#ifndef SYSCALL_ASM
#define SYSCALL_ASM

#include "types.h"

//goes through enable_paging routine for page_dir
extern void context_switch(uint32_t ss, uint32_t esp, uint32_t cs, uint32_t eip);

extern void return_from_halt(uint32_t ebp, uint32_t status);


#endif
