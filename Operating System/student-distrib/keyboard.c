#include "keyboard.h"
#include "schedule.h"
volatile static int char_count = 0;
volatile static int first_char = 1;
volatile static int caps_lock_flag = 0;
volatile static int control_flag = 0;
volatile static int l_shift_flag = 0;
volatile static int r_shift_flag = 0;
volatile static int shift_flag = 0;
volatile static int alt_flag = 0;

/* void init_keyboard()
 *
 * initalizes keyboard
 * Inputs: None
 * Outputs: Void
 * Side Effects: keyboard activated
 */
void init_keyboard(){
    setEnterFlag(0);
    os_flag = 1;
    starting_flag = 1;
    keyboard_read = 0;
    pop_keyboard();
    enable_irq(0x01);
}


/* void keypress()
 *
 * receives and puts pressed character onto the screen
 * Inputs: None
 * Outputs: Void
 * Side Effects: screen displays character
 */
void keypress(){ //return value was 1
    cli();

    uint8_t keycode;
    keycode = inb(0x60); //keyboard port

    //Avoids removing non-existent character.
    if(keycode == BACKSPACE && terminals[active_terminal].cur_idx > 0){
        
        terminals[active_terminal].cur_idx--; // decrement characters stored in buffer
        keyboard_buf[terminals[active_terminal].cur_idx] = '\0';
        
        putc_nos(ASCII_BACKSPACE);
        if(char_count==0) {
            char_count = 80;
        }
        else {
            char_count--;
        }
    }
    // Set Shift Flags
    else if(keycode == L_SHIFT_ON) {
        l_shift_flag = 1;
        shift_flag = ~shift_flag;
    }
    else if(keycode == L_SHIFT_OFF) {
        l_shift_flag = 0;
        shift_flag = ~shift_flag;
    }
    else if(keycode == R_SHIFT_ON) {
        r_shift_flag = 1;
        shift_flag = ~shift_flag;
    }
    else if(keycode == R_SHIFT_OFF) {
        r_shift_flag = 0;
        shift_flag = ~shift_flag;
    }
    // Set Caps Flag
    else if(keycode == CAPS_LOCK_ON) {
        if(caps_lock_flag) {
            caps_lock_flag = 0;
        }
        else {
            caps_lock_flag = 1;
        }
    }
    // Set Control Flags
    else if(keycode == CONTROL_ON) {
        control_flag = 1;
    }
    else if(keycode == CONTROL_OFF) {
        control_flag = 0;
    }
    else if(keycode == ALT_ON){
        alt_flag = 1;
    }
    else if(keycode == ALT_OFF){
        alt_flag = 0;
    }
    else if(keycode == ENTER){
        putc_nos('\n');
        setEnterFlag(1);
        
        char_count = 0;
        terminals[get_active()].cur_idx = 0; //reset everything
    }
    else if (keycode == ENTER_RELEASE){
        memset(keyboard_buf, 0, BUFFER_SIZE-1);
    }
    else if(control_flag && convert[keycode]=='l') {
        char_count = 0;
        clear();
        reset_screen();
    }
    else if(alt_flag && (keycode == F1 || keycode == F2 || keycode == F3)){
        switch_terminal(keycode - 0x3B);
    }
    else{
        if(convert[keycode] != '\0') {
            if(terminals[active_terminal].cur_idx < BUFFER_SIZE-1) {
                if(caps_lock_flag && shift_flag){
                    if(convert[keycode] >= 'a' && convert[keycode] <= 'z'){
                        putc_nos(convert[keycode]);
                        keyboard_buf[terminals[active_terminal].cur_idx] = convert[keycode]; //if letter and caps+shift then lowercase
                    }
                    else{
                        putc_nos(convert_shift[keycode]);
                        keyboard_buf[terminals[active_terminal].cur_idx] = convert_shift[keycode]; //else shift
                    }
                    
                }
                else if(caps_lock_flag || shift_flag) {
                    if(convert[keycode] >= 'a' && convert[keycode] <= 'z'){
                        putc_nos(convert_shift[keycode]); 
                        keyboard_buf[terminals[active_terminal].cur_idx] = convert_shift[keycode]; //if letter and caps xor shift then upper
                    }
                    else{
                        if(shift_flag){ 
                            putc_nos(convert_shift[keycode]);
                            keyboard_buf[terminals[active_terminal].cur_idx] = convert_shift[keycode]; //if not letter, but only shift pressed, then shift
                        }
                        else{
                            putc_nos(convert[keycode]);
                            keyboard_buf[terminals[active_terminal].cur_idx] = convert[keycode]; //not letter and caps, so no change
                        }
                    }
                    
                }
                else {
                    putc_nos(convert[keycode]);
                    keyboard_buf[terminals[active_terminal].cur_idx] = convert[keycode]; //no caps/shift flag, just put char
                }
                char_count++;
                if(char_count == 80 - SHELL_LENGTH) { // NUM_COLS = 80
                    char_count = 0;
                    putc_nos('\n');
                }
                terminals[active_terminal].cur_idx++;
            }
        }
    }
    
    send_eoi(0x01); //used to return 1, 1 is the keyboard irq number
    
}

/* void pop_keyboard()
 *
 * initalizes convert scan codes to ascii
 * Inputs: None
 * Outputs: Void
 * Side Effects: convert ready, convert_shift ready
 */
void pop_keyboard(){

    convert[0] = '\0'; //used to be '@' - likely accidental but change made here
    convert[0x02] = '1';
    convert[0x03] = '2';
    convert[0x04] = '3';
    convert[0x05] = '4';
    convert[0x06] = '5';
    convert[0x07] = '6';
    convert[0x08] = '7';
    convert[0x09] = '8';
    convert[0x0A] = '9';
    convert[0x0B] = '0';
    convert[0x0C] = '-';
    convert[0x0D] = '=';
    convert[0x0F] = ASCII_TAB;
    convert[0x10] = 'q';
    convert[0x11] = 'w';
    convert[0x12] = 'e';
    convert[0x13] = 'r';
    convert[0x14] = 't';
    convert[0x15] = 'y';
    convert[0x16] = 'u';
    convert[0x17] = 'i';
    convert[0x18] = 'o';
    convert[0x19] = 'p';
    convert[0x1A] = '[';
    convert[0x1B] = ']';
    convert[0x1E] = 'a';
    convert[0x1F] = 's';
    convert[0x20] = 'd';
    convert[0x21] = 'f';
    convert[0x22] = 'g';
    convert[0x23] = 'h';
    convert[0x24] = 'j';
    convert[0x25] = 'k';
    convert[0x26] = 'l';
    convert[0x27] = ';';
    convert[0x28] = 0x27; //ascii for single quote
    convert[0x29] = '`';
    convert[0x2B] = 0x5C; //ascii for backslash
    convert[0x2C] = 'z';
    convert[0x2D] = 'x';
    convert[0x2E] = 'c';
    convert[0x2F] = 'v';
    convert[0x30] = 'b';
    convert[0x31] = 'n';
    convert[0x32] = 'm';
    convert[0x33] = ',';
    convert[0x34] = '.';
    convert[0x35] = '/';
    convert[0x39]= ' ';

    
    convert_shift[0x02] = '!';
    convert_shift[0x03] = '@';
    convert_shift[0x04] = '#';
    convert_shift[0x05] = '$';
    convert_shift[0x06] = '%';
    convert_shift[0x07] = '^';
    convert_shift[0x08] = '&';
    convert_shift[0x09] = '*';
    convert_shift[0x0A] = '(';
    convert_shift[0x0B] = ')';
    convert_shift[0x0C] = '_';
    convert_shift[0x0D] = '+';
    convert_shift[0x0F] = ASCII_TAB;
    convert_shift[0x10] = 'Q';
    convert_shift[0x11] = 'W';
    convert_shift[0x12] = 'E';
    convert_shift[0x13] = 'R';
    convert_shift[0x14] = 'T';
    convert_shift[0x15] = 'Y';
    convert_shift[0x16] = 'U';
    convert_shift[0x17] = 'I';
    convert_shift[0x18] = 'O';
    convert_shift[0x19] = 'P';
    convert_shift[0x1A] = '{';
    convert_shift[0x1B] = '}';
    convert_shift[0x1E] = 'A';
    convert_shift[0x1F] = 'S';
    convert_shift[0x20] = 'D';
    convert_shift[0x21] = 'F';
    convert_shift[0x22] = 'G';
    convert_shift[0x23] = 'H';
    convert_shift[0x24] = 'J';
    convert_shift[0x25] = 'K';
    convert_shift[0x26] = 'L';
    convert_shift[0x27] = ':';
    convert_shift[0x28] = 0x22; //ascii for "
    convert_shift[0x29] = '~';
    convert_shift[0x2B] = '|';
    convert_shift[0x2C] = 'Z';
    convert_shift[0x2D] = 'X';
    convert_shift[0x2E] = 'C';
    convert_shift[0x2F] = 'V';
    convert_shift[0x30] = 'B';
    convert_shift[0x31] = 'N';
    convert_shift[0x32] = 'M';
    convert_shift[0x33] = '<';
    convert_shift[0x34] = '>';
    convert_shift[0x35] = '?';
    convert_shift[0x39]= ' ';
}

