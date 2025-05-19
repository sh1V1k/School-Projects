#pragma once
#include "lib.h"
#include "i8259.h"
#include "schedule.h"

#define A 0x8A
#define B 0x8B
#define C 0x8C
#define RTC_PORT 0x70
#define CMOS_PORT 0x71
#define DEFAULT_RATE 1024

int rtc_count[3];
int magic_rate[3]; //used to make it appear we have a slower interrupt rate
int waiting[3]; //used for rtc read

extern void init_rtc();
extern void rtc_interrupt();
extern int32_t rtc_open(const uint8_t *fname);
extern int32_t rtc_read(int32_t fd, void *buf, int32_t nbytes);
extern int32_t rtc_write(int32_t fd, const void *buf, int32_t nbytes);
extern int32_t rtc_close(int32_t fd);

typedef struct file_opertation_rtc{ //jump table for function pointers
  int32_t (*rtc_open)(const uint8_t *fname);
  int32_t (*rtc_read)(int32_t fd, void *buf, int32_t nbytes);
  int32_t (*rtc_close)(int32_t fd); 
  int32_t (*rtc_write)(int32_t fd, const void *buf, int32_t nbytes);
}for_t;

