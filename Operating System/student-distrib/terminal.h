#pragma once
#ifndef TERMINAL_H
#define TERMINAL_H
#define BUFFER_SIZE 128
#define BACKSPACE 0x0E
#define ASCII_BACKSPACE 0x08
#define CAPS_LOCK_ON 0x3A
#define CONTROL_ON 0x1D
#define CONTROL_OFF 0x9D
#define L_SHIFT_ON 0x2A
#define L_SHIFT_OFF 0xAA
#define R_SHIFT_ON 0x36
#define R_SHIFT_OFF 0xB6
#define ALT_ON 0x38
#define ALT_OFF 0xB8
#define F1 0x3B
#define F2 0x3C
#define F3 0x3D
#define ENTER 0x1C
#define ENTER_RELEASE 0x9C
#define TAB_PRESSED 0x0F
#define ASCII_TAB 0x09
#define NUM_COLS 80

#include "types.h"
#include "lib.h"
#include "schedule.h"
#include "keyboard.h"

int32_t read_terminal(int32_t fd, void* buffer, int32_t number_bytes);
int32_t write_terminal(int32_t fd, const void* buf, int32_t number_bytes);
int32_t open_terminal(const uint8_t* filename);
int32_t close_terminal(int32_t fd);

void setEnterFlag(uint32_t val);
void setKeyRead(uint32_t val);

extern void get_char(char c);
char terminal_buffer[3][BUFFER_SIZE];
volatile static int starting_flag = 1;
volatile static int os_flag = 1;
volatile static int write_next = 0;
volatile static int enter_flag = 0;
volatile static int keyboard_read;

#endif
