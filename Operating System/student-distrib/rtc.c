#include "rtc.h"
 #include "lib.h"

/* void init_rtc()
 *
 * Initalizes the Real-Time Clock
 * Inputs: None
 * Outputs: Void
 * Side Effects: RTC activated
 */
void init_rtc(){
    printf("starting rtc initialization \n");
    unsigned char prev;
    outb(A, RTC_PORT);	// select Status Register A, and disable NMI (by setting the 0x80 bit)
    
    outb(B, RTC_PORT);
    prev = inb(CMOS_PORT);
    outb(B, RTC_PORT);
    outb(prev | 0x40, CMOS_PORT);
    enable_irq(0x8);
    rtc_count[0] = 0; // init ticks
    rtc_count[1] = 0; // init ticks
    rtc_count[2] = 0; // init ticks
    printf("done init rtc\n");
    
}
/* void rtc_interrupt()
 *
 * Recieve RTC Interrupt 
 * Inputs: None
 * Outputs: Void
 * Side Effects: RTC activated; rtc_count incremented
 */
void rtc_interrupt(){
    //clear();
    //test_interrupts();
    outb(C,RTC_PORT);
    inb(CMOS_PORT);

    rtc_count[0]++; //ticks ++
    rtc_count[1]++; //ticks ++
    rtc_count[2]++; //ticks ++
    send_eoi(0x8);
}


// FOR CHECKPOINT 2 BELOW
int32_t rtc_open(const uint8_t *fname) {
    magic_rate[get_process()] = DEFAULT_RATE;
    return 0;
}

int32_t get_power_of_two(int32_t value) {
    unsigned count = 0;
    if(value <= 0){ return -1; } //invalid power of two rate also would get divide by zero error
    while(value != 1) {
        value = value >> 1;
        count++;
    }
    
    return count;
}

int32_t rtc_write(int32_t fd, const void* buf, int32_t nbytes) {
    uint32_t new_freq;
    new_freq = *(uint32_t*)buf;
    // new_freq = 1024;
    if(get_power_of_two(new_freq) != -1){
        magic_rate[get_process()] = DEFAULT_RATE/new_freq;
        rtc_count[get_process()] = 0;
        return 0;
    } else{ return -1; } //invalid freq
}

int32_t rtc_close(int32_t fd) {
    magic_rate[get_process()] = DEFAULT_RATE;
    rtc_count[get_process()] = 0; // init ticks
    return 0;
}

//need to update to account for virtualization, basically should only return 0 depending
//on rtc count and the frequency that has been set by write
int32_t rtc_read(int32_t fd, void *buf, int32_t nbytes) {
    uint32_t process = get_process();
    waiting[process] = 1;
    rtc_count[process] = 0;
    while(waiting[process]){
        if(rtc_count[process] >= magic_rate[process]){
            waiting[process] = 0;
        }
        
    }
    return 0;
}
