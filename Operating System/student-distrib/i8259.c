/* i8259.c - Functions to interact with the 8259 interrupt controller
 * vim:ts=4 noexpandtab
 */

#include "i8259.h"
#include "lib.h"

/* Interrupt masks to determine which interrupts are enabled and disabled */
uint8_t master_mask; /* IRQs 0-7  */
uint8_t slave_mask;  /* IRQs 8-15 */

/* Initialize the 8259 PIC */
/* i8259_init()
 * 
 * Initialize PIC
 * Inputs: None
 * Outputs: NONE
 * Side Effects: Initialized PIC
 */
void i8259_init(void) {

    //unsigned long flags;
    //cli_and_save(flags);
    printf("Initializing PIC\n");
    
    //mask all interrupts
    outb(0xFF, MASTER_8259_PORT+1);
    outb(0xFF, SLAVE_8259_PORT+1); 

    //ICW1
    // maybe use outb_p below
    outb(ICW1, MASTER_8259_PORT);
    outb(ICW1, SLAVE_8259_PORT);

    //ICW2
    outb(ICW2_MASTER, MASTER_8259_PORT+1);
    outb(ICW2_SLAVE, SLAVE_8259_PORT+1);
    
    //ICW3
    outb(ICW3_MASTER, MASTER_8259_PORT+1);
    outb(ICW3_SLAVE, SLAVE_8259_PORT+1);

    //ICW4
    outb(ICW4, MASTER_8259_PORT+1);
    outb(ICW4, SLAVE_8259_PORT+1);
    
    outb(0xFF, MASTER_8259_PORT+1);
    outb(0xFF, SLAVE_8259_PORT+1);
         
    enable_irq(2);

    printf("done init PIC\n");

}

/* Enable (unmask) the specified IRQ */
/* enable_irq(irq_num)
 * 
 * Enable a specific irq
 * Inputs: irq_num - irq to enable
 * Outputs: NONE
 * Side Effects: Enables the given irq
 */
void enable_irq(uint32_t irq_num) {
    if(irq_num < 0 || irq_num > 16){ return; } //bounds check
    uint32_t port;
    uint32_t value;
    unsigned long flags;
    cli_and_save(flags);

    if(irq_num < 0 || irq_num > 15){ return;}

    //decide which pic the interrupt came from
    if(irq_num < 8){ port = MASTER_8259_PORT + 1; }
    else{
        port = SLAVE_8259_PORT + 1;
        irq_num -= 8; //update irq_num to be 0-7 for slave PIC
    }
    
    value = inb(port) & ~(1 << irq_num); //set mask
    outb(value, port);
    restore_flags(flags);
}

/* Disable (mask) the specified IRQ */
/* disable_irq(irq_num)
 * 
 * Disable a specific irq
 * Inputs: irq_num - irq to disable
 * Outputs: NONE
 * Side Effects: Disables the given irq
 */
void disable_irq(uint32_t irq_num) {
    if(irq_num < 0 || irq_num > 16){ return; } //bounds check
    uint32_t port;
    uint32_t value;

    unsigned long flags;
    cli_and_save(flags);

    //decide which pic the interrupt came from
    if(irq_num < 8){ port = MASTER_8259_PORT + 1; }
    else{
        port = SLAVE_8259_PORT + 1;
        irq_num -= 8; //update irq_num to be 0-7 for slave PIC
    }
    
    value = inb(port) | (1 << irq_num); //clear mask, had a ~(1 << irq_num) before
    outb(value, port);

    restore_flags(flags);
}

/* Send end-of-interrupt signal for the specified IRQ */
/* send_eoi(irq_num)
 * 
 * Send eoi for specified irq
 * Inputs: irq_num - irq to send eoi for
 * Outputs: NONE
 * Side Effects: sends eoi for specified irq
 */
void send_eoi(uint32_t irq_num) {
    if(irq_num < 0 || irq_num > 16){ return; } //bounds check
    unsigned long flags;
    cli_and_save(flags);
    if(irq_num >= 8){ 
        outb(EOI | (irq_num & 7), SLAVE_8259_PORT);
        outb(EOI | 2, MASTER_8259_PORT);
    } //if irq came from slave give it eoi as well
    outb(EOI | irq_num, MASTER_8259_PORT);
    restore_flags(flags);
}
