#pragma once
#ifndef KEYBOARD_H
#define KEYBOARD_H
#define KEYBOARD_SIZE 256
#define SHELL_LENGTH 7

#include "lib.h"
#include "i8259.h"
#include "terminal.h"

typedef struct code_conversion{
    uint16_t keycode;
    char character;  
} code_conversion;

char convert[KEYBOARD_SIZE];
char convert_shift[KEYBOARD_SIZE];
void pop_keyboard();
extern void init_keyboard();
extern void keypress(); //return type was int and returned 1
void clear_buffer();
char keyboard_buf[BUFFER_SIZE-1];
//need a keyboard interrupt
#endif

