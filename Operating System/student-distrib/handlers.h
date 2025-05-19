#pragma once
#ifndef HANDLER_H
#define HANDLER_H

#include "lib.h"
extern void divide_error();
extern void debug();
extern void nmi();
extern void breakpoint();
extern void overflow();
extern void bounds();
extern void opcode();
extern void coprocessor();
extern void double_fault();
extern void segment_overrun();
extern void invalid_tss();
extern void seg_not_present();
extern void stack_fault();
extern void general_protection_fault();
extern void page_fault();
extern void math_fault();
extern void alignment_check();
extern void machine_check();
extern void simd_math_fault();
extern void virtualization_fault();
//extern void rtc_int();
//extern void syscall();
//extern void keyboard_int();
extern void KEYBOARD_INT_MACRO();
extern void RTC_INT_MACRO();
extern void PIT_INT_MACRO();

#endif

