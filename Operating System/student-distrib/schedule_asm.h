#pragma once
#ifndef SCHEDULE_ASM
#define SCHEDULE_ASM

#include "types.h"

extern void process_context(uint32_t ss, uint32_t esp, uint32_t cs, uint32_t eip);

extern void return_schedule(uint32_t ebp, uint32_t esp);


#endif
