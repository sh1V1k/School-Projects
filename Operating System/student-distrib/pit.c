#include "pit.h"


//divide by 30,000(reload value) to get 39.77 hz ~ 25 ms
//should use mode 2 since i wont have to set reload value each time


/*
Shivam notes !
Typically, OSes and BIOSes use mode 3 (see below) for PIT channel 0 to generate IRQ 0 timer ticks,
but some use mode 2 instead, to gain frequency accuracy (frequency = 1193182 / reload_value Hz).

18-1193181 hz range for pit
0-65536 range for reload values
*/

void pit_init(){
    // cli();
    
    outb(0x34, MODE_CMD_REG); //channel 0, lobyte/hibyte, rate generator

    //want to put reload value at channel 0 (need to do it in two steps, low byte and high byte separately
    outb((RELOAD_VAL & 0xFF), CHANNEL_0); //low byte
    outb((RELOAD_VAL >> 8), CHANNEL_0); //high byte
    enable_irq(0);
    // sti();
}

