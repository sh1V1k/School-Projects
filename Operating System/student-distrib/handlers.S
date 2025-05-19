#define ASM 1

// EXCEP_NO_ERR(function_name, vector)
// function_name - name of function to define
// vector - interrupt vector
// creates handler for every interrupt and call common handler, is made for interrupts with no error code

#define EXCEP_NO_ERR(function_name, vector) \
    .global function_name ;\
    function_name:        ;\
        pushl $0              ;\
        pushal                ;\
        pushfl                ;\
        pushl $vector         ;\
        call handler          ;\
        addl $4, %esp         ;\
        popfl                 ;\
        popal                 ;\
        addl $4, %esp         ;\
        iret                  ;\

// EXCEP_WITH_ERR(function_name, vector)
// function_name - name of function to define
// vector - interrupt vector
// creates handler for every interrupt and call common handler, is made for interrupts with with error code


#define EXCEP_WITH_ERR(function_name, vector) \
    .global function_name ;\
    function_name:        ;\
        pushal                ;\
        pushfl                ;\
        pushl $vector         ;\
        call handler          ;\
        addl $4, %esp         ;\
        popfl                 ;\
        popal                 ;\
        iret                  ;\

#define DEVICE_HANDLER(name, handler)        \
    .globl name                             ;\
    name:                                   ;\
        pushal                              ;\
        call handler                        ;\
        popal                               ;\
        iret                                ;\


EXCEP_NO_ERR(divide_error, 0);
EXCEP_NO_ERR(debug, 1);
EXCEP_NO_ERR(nmi, 2);
EXCEP_NO_ERR(breakpoint, 3);
EXCEP_NO_ERR(overflow, 4);
EXCEP_NO_ERR(bounds, 5);
EXCEP_NO_ERR(opcode, 6);
EXCEP_NO_ERR(coprocessor, 7);
EXCEP_WITH_ERR(double_fault, 8); 
EXCEP_NO_ERR(segment_overrun, 9);
EXCEP_WITH_ERR(invalid_tss, 10);
EXCEP_WITH_ERR(seg_not_present, 11);
EXCEP_WITH_ERR(stack_fault, 12);
EXCEP_WITH_ERR(general_protection_fault, 13);
EXCEP_WITH_ERR(page_fault, 14);
EXCEP_NO_ERR(math_fault, 16);
EXCEP_WITH_ERR(alignment_check, 17);
EXCEP_NO_ERR(machine_check, 18);
EXCEP_NO_ERR(simd_math_fault, 19);
EXCEP_NO_ERR(virtualization_fault, 20);

DEVICE_HANDLER(KEYBOARD_INT_MACRO, keypress);
DEVICE_HANDLER(RTC_INT_MACRO, rtc_interrupt);
DEVICE_HANDLER(PIT_INT_MACRO, schedule);
