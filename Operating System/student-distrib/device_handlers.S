#define ASM

#define DEVICE_HANDLER(name, handler)       \
    .globl name                             ;\
    name:                                   ;\
        pushal                              ;\
        call handler                        ;\
        popal                               ;\
        iret                                ;\


DEVICE_HANDLER(KEYBOARD_MACRO, keypress);
DEVICE_HANDLER(RTC_MACRO, rtc_interrupt);
