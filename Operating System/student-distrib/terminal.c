#include "terminal.h"

//GLOBAL
volatile static int new_line_flag = 0;
volatile static int character_count = 0;
volatile int global_col_index = 0;
volatile int CLEAR_THIS_BUFFER = 0;



/* uint32_t read_terminal(int32_t fd, void* buffer, uint32_t number_bytes);
 *   Inputs: fd, buffer, num__bytes
 *   Return Value: Number of bytes written
 *    Function: Copies keyboard buffer to terminal buffer */
int32_t read_terminal(int32_t fd, void* buffer, int32_t number_bytes) {
    //unsigned index;
    int process_screen = get_process();
    //int active_screen = get_active();
    
    int threshold = number_bytes;

    if(BUFFER_SIZE - 1 < threshold) {
        threshold = BUFFER_SIZE - 1;
    } //ensure number of bytes read is within range
    // if(threshold < BUFFER_SIZE-1) threshold++;
    sti();
    while(!terminals[process_screen].read_flag){}
    terminals[process_screen].read_flag = 0;
    strncpy(terminals[process_screen].terminal_buffer, keyboard_buf, BUFFER_SIZE-1);
    strncpy((char*)buffer, terminals[process_screen].terminal_buffer, number_bytes);

    return threshold;
}

/* uint32_t write_terminal(int32_t fd, const void* buf, uint32_t number_bytes);
 *   Inputs: fd, buf, num__bytes
 *   Return Value: Number of bytes written to terminal
 *    Function: Writes terminal buffer to terminal */
int32_t write_terminal(int32_t fd, const void* buf, int32_t number_bytes) {
    cli();
    char c;
    unsigned index;
    unsigned count = 0;
    int threshold = number_bytes;
    character_count = 0;
    //ensure number of bytes written is within size of buffer
    for(index = 0; index < threshold; index++) {
        c = ((char*)buf)[index];
        // Not Null Character or smiley face char
        if(c != '\0' && c!= '\002') {
            putc(c);
            count++;
        }
        global_col_index++;
    }
    sti();
    return count;
}

/* uint32_t open_terminal(const uint8_t* filename);
 *   Inputs: filename
 *   Return Value: 0
 *    Function: Opens terminal */
int32_t open_terminal(const uint8_t* filename) {
    return -1;
}

/* uint32_t close_terminal(int32_t fd);
 *   Inputs: fd
 *   Return Value: 0
 *    Function: Closes terminal
 */
int32_t close_terminal(int32_t fd) {
    return -1;
}
/* uint32_t setEnterFlag(int32_t val);
 *   Inputs: val
 *   Return Value: void
 *   Function: sets the enter_flag
 */
void setEnterFlag(uint32_t val){ 
    terminals[get_active()].read_flag = val;
}
/* uint32_t setKeyRead(int32_t val);
 *   Inputs: val
 *   Return Value: void
 *   Function: sets the keyboard_read
 */
void setKeyRead(uint32_t val){
    keyboard_read = val;
}

