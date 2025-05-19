#pragma once
#ifndef _PIT_H
#define _PIT_H

#include "types.h"
#include "lib.h"
#include "i8259.h"

#define CHANNEL_0 0x40
#define CHANNEL_1 0x41
#define CHANNEL_2 0x42
#define MODE_CMD_REG 0x43
#define RELOAD_VAL 0x7530

void pit_init();


#endif 
